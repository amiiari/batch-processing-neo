"""
Batch Hires-Fix tab
===================
A UI tab where you drag in multiple txt2img-generated images and run each
through hires-fix in batch, reusing Forge's existing pipeline. Runs SECOND in
the refine chain, after Batch ADetailer — the scan/save plumbing is the shared
core parameterized with this stage's record.
"""
import importlib
import json
import os
import re
import time
import traceback
from contextlib import closing

import gradio as gr
from PIL import Image

from modules import call_queue, images, processing, script_callbacks, scripts, shared
from modules_forge import main_thread

import batch_adetailer_shared as bshared
# scripts/*.py are re-executed on every in-process Reload UI, but root modules
# stay cached in sys.modules — reload so shared-code edits land too.
importlib.reload(bshared)

STAGE = bshared.HIRES_STAGE

# ──────────────────────────────────────────────
# Extension Settings (registered via add_option)
# ──────────────────────────────────────────────
def _register_settings():
    section = ("batch_hires_fix", "Batch Hires-Fix")

    shared.opts.add_option(
        "batch_hires_fix_output_dir",
        shared.OptionInfo("", "Output Directory", gr.Textbox, {}, section=section)
        .info("Leave empty to use the default txt2img output directory."),
    )

    shared.opts.add_option(
        "batch_hires_fix_scan_roots",
        shared.OptionInfo(
            "",
            "Test-folder scan roots (semicolon-separated)", gr.Textbox, {}, section=section)
        .info("Each root is searched (up to 3 levels deep) for Tests folders — so one "
              "root covers both <root>/<set>/Tests and <root>/Commissions/<set>/Tests. "
              "The same roots are used to find drag-dropped images back on disk. "
              "Non-existent roots are silently skipped."),
    )

    shared.opts.add_option(
        "batch_hires_fix_skip_errors",
        shared.OptionInfo(True, "Skip Failed Images and Continue", gr.Checkbox,
                          {}, section=section)
        .info("If an image fails during hires-fix, skip it and continue with the rest."),
    )

# ──────────────────────────────────────────────
# Test-folder scanning — shared core, parameterized by this stage's record.
#
# Hires-fix runs first: it picks up base images (NrM.png) that have no -hires
# successor and saves the hires-fix result as <stem>-hires.png, which Batch
# ADetailer then takes. A plain Lanczos upscale (<stem>-base.png) at the same
# resolution is saved too only when that box is ticked (off by default).
# ──────────────────────────────────────────────
def _base_images(folder):
    return bshared.stage_inputs(folder, STAGE)


def _pending_bases(folder):
    return bshared.pending_inputs(folder, STAGE)


# ──────────────────────────────────────────────
# Per-image prompt overrides
#
# The hires pass is conditioned on exactly ONE prompt: firstpass_image skips the
# first pass entirely (processing.py :: StableDiffusionProcessingTxt2Img.init),
# and hr_prompt="" makes it fall back to p.prompt. So an override is just "what
# p.prompt should be for this image".
#
# The store is {path: prompt}. A path with no entry runs with the prompt read
# from its own infotext, exactly as before — so membership, not truthiness,
# decides: an entry of "" is a real override (hires with no prompt).
# ──────────────────────────────────────────────
# `<n>r<rev>` with any pipeline suffix: 20r3, 20r3-adetailer, 20r3-adetailer-base.
_REVISION_RE = re.compile(r"^(\d+)r(\d+)(?:-.+)?$", re.IGNORECASE)


def _prompt_for(store, path):
    store = store or {}
    return str(store[path]) if path in store else bshared.image_prompt(path)


def _first_revision_prompt(path):
    """Prompt from the lowest-numbered revision of the same image in the same
    folder: 20r3-adetailer -> the prompt baked into 20r1 (or, failing a plain
    20r1, whichever 20r1 variant is on disk). Returns (prompt_or_None, note).

    Only the *latest* revision of each image is ever loaded into this tab, so the
    earlier ones sit on disk but never in the gallery — hence reading them here.
    """
    m = _REVISION_RE.match(os.path.splitext(os.path.basename(path))[0])
    if not m:
        return None, "that filename isn't `<number>r<revision>` — no earlier revision to read."

    number = m.group(1)
    by_rev: dict = {}
    for stem, fname in bshared.image_stems(os.path.dirname(path)).items():
        rm = _REVISION_RE.match(stem)
        # String-compare the number so 020r1 and 20r1 stay different images.
        if rm and rm.group(1) == number:
            by_rev.setdefault(int(rm.group(2)), []).append((stem, fname))
    if not by_rev:
        return None, "no revisions of that image found next to it."

    rev = min(by_rev)
    if rev >= int(m.group(2)):
        return None, f"this already *is* revision {rev}."

    # The plain `<n>r<rev>` base first — it carries the original generation's
    # infotext; its -adetailer/-hires variants are the fallback.
    folder = os.path.dirname(path)
    for stem, fname in sorted(by_rev[rev],
                              key=lambda sf: ("-" in sf[0], bshared.natural_key(sf[0]))):
        prompt = bshared.image_prompt(os.path.join(folder, fname))
        if prompt:
            return prompt, f"loaded the prompt from `{fname}`."
    return None, f"revision {number}r{rev} exists, but none of its files carry a prompt."


# ──────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────
def _save_base_copy(img: Image.Image, geninfo: str | None, stem: str, outdir: str, size):
    """
    Plain Lanczos upscale of the source at the hires result's exact size,
    saved as <stem>-base.png next to it — the image with no hires pass, for
    comparing or layering by hand. Carries the source's generation
    info. Skipped if it already exists (re-runs stay idempotent).
    """
    dest = os.path.join(outdir, f"{stem}-base.png")
    if os.path.exists(dest):
        return
    from PIL.PngImagePlugin import PngInfo
    meta = PngInfo()
    if geninfo:
        meta.add_text("parameters", geninfo)
    img.resize(size, Image.LANCZOS).save(dest, pnginfo=meta)


# ──────────────────────────────────────────────
# Core Processing Logic — mirrors txt2img_upscale_function
# ──────────────────────────────────────────────
def _process_single_image(img: Image.Image, geninfo: str | None, hires_params: dict,
                          save_opts: dict, prompt_override: str | None = None):
    """
    Process one image through hires-fix. Mirrors the logic in
    modules/txt2img.py :: txt2img_upscale_function().

    Key difference from the ✨ button: we do NOT set p.txt2img_upscale = True,
    because that flag causes Forge Neo to skip model loading (it assumes the ✨
    button already has a loaded model). Since our extension runs independently,
    we need normal model loading to occur inside process_images().

    Runs on Forge's main thread (see batch_hires_fix_process). Never raises:
    returns (images, infotexts, error_traceback_or_None, notes) so the full
    traceback reaches the status log instead of being swallowed by Gradio.
    """
    notes: list[str] = []
    try:
        p = processing.StableDiffusionProcessingTxt2Img(
            outpath_samples=(
                save_opts.get("output_dir")
                or getattr(shared.opts, "batch_hires_fix_output_dir", None)
                or shared.opts.outdir_samples
                or shared.opts.outdir_txt2img_samples
            ),
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
            prompt="",
            styles=[],
            negative_prompt="",
            batch_size=1,
            n_iter=1,
            # Placeholders: apply_source_image_parameters sets the real ones.
            cfg_scale=7.0,
            distilled_cfg_scale=3.0,
            width=img.size[0],
            height=img.size[1],
            enable_hr=True,
            denoising_strength=float(hires_params.get("denoising_strength", 0.6)),
            hr_scale=float(hires_params.get("hr_scale", 2.0)),
            hr_upscaler=hires_params.get("hr_upscaler"),
            hr_second_pass_steps=int(hires_params.get("hr_second_pass_steps", 0)),
            hr_resize_x=int(hires_params.get("hr_resize_x", 0)),
            hr_resize_y=int(hires_params.get("hr_resize_y", 0)),
            hr_checkpoint_name=None,
            hr_additional_modules=["Use same choices"],
            hr_sampler_name=(
                None if hires_params.get("hr_sampler_name") == "Use same sampler"
                else hires_params.get("hr_sampler_name")
            ),
            hr_scheduler=(
                None if hires_params.get("hr_scheduler") == "Use same scheduler"
                else hires_params.get("hr_scheduler")
            ),
            hr_prompt="",
            hr_negative_prompt="",
            hr_cfg=float(hires_params.get("hr_cfg", 6.0)),
            hr_distilled_cfg=3.0,  # replaced by the source shift below
            override_settings={},
        )

        # Same pattern as txt2img_create_processing: assign scripts + real
        # default args and let the setters run setup_scripts() normally.
        p.scripts = scripts.scripts_txt2img
        p.script_args = bshared.get_default_script_args(scripts.scripts_txt2img, "txt2img").copy()

        if geninfo:
            # The ✨ button gets these for free from the live txt2img UI state;
            # we must recover them from the image's infotext.
            params = bshared.apply_source_image_parameters(p, geninfo)
            # Scripts that can rebuild their settings from the image's own (Stagehand's
            # Precise Reference: its reference images) get them instead of UI defaults.
            bshared.replay_script_args(scripts.scripts_txt2img, p.script_args, params)
            # Shift-based models (Qwen, Flux, ...): use the same shift for the
            # hires pass as the original generation, like "Use same" semantics.
            p.hr_distilled_cfg = p.distilled_cfg_scale

        if prompt_override is not None:
            # This image's edited prompt. hr_prompt stays "" so the hires pass
            # reuses p.prompt; p.styles (recovered from the infotext) are still
            # folded in on top, exactly as they would be without an override.
            p.prompt = prompt_override

        # An old image's prompt can name a LoRA that no longer exists under
        # that name — the hires pass would silently render without it.
        p.prompt, found = bshared.fix_lora_names(p.prompt)
        notes += found
        p.negative_prompt, found = bshared.fix_lora_names(p.negative_prompt)
        notes += found
        for note in dict.fromkeys(notes):
            print(f"[Batch Hires-Fix] {note}")

        p.firstpass_image = img
        # Intentionally NOT setting p.txt2img_upscale — see docstring above.

        p.override_settings["save_images_before_highres_fix"] = False

        if save_opts.get("use_original_name") or save_opts.get("discard"):
            # Saved by hand afterwards with the original filename + suffix -- or, for a
            # forge link slot's image, not here at all: the slot saves it.
            p.do_not_save_samples = True

        with closing(p):
            processed = scripts.scripts_txt2img.run(p, *p.script_args)

            if processed is None:
                processed = processing.process_images(p)

        if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
            # A cancelled sampling loop returns the partially-denoised image;
            # saving it would make the result look finished (and folder mode
            # would then never re-list the base) — drop it instead.
            return [], [], None, notes

        if save_opts.get("use_original_name"):
            bshared.save_with_original_name(processed, p, save_opts)

        if save_opts.get("base_copy") and processed.images:
            # The Krita edit stage layers -hires over an unedited upscale of
            # the same size; a twin failure shouldn't fail the image.
            try:
                _save_base_copy(img, geninfo, save_opts["stem"],
                                p.outpath_samples, processed.images[0].size)
            except Exception as e:
                print(f"[Batch Hires-Fix] -base copy failed: {e}")

        return processed.images, processed.infotexts, None, notes
    except Exception:
        tb = traceback.format_exc()
        print(f"[Batch Hires-Fix] Error processing image:\n{tb}")
        return [], [], tb, notes


def batch_hires_fix_process(
    files,
    denoising_strength,
    hr_scale,
    hr_upscaler,
    hr_second_pass_steps,
    hr_resize_x,
    hr_resize_y,
    hr_sampler_name,
    hr_scheduler,
    hr_cfg,
    use_original_name,
    filename_suffix,
    save_base_copy=False,
    save_to_source=False,
    prompt_store=None,
    sel=None,
    live_prompt=None,
    char_store=None,
):
    """
    Main batch processing function. Processes each image through hires-fix
    sequentially and collects all results.

    save_base_copy: also save a plain Lanczos upscale of each source as
    <stem>-base.png at the result's resolution (no hires pass; off by default).
    save_to_source: each result is saved into the directory its source image
    came from (folder mode), instead of the configured output directory.
    prompt_store/sel/live_prompt: the per-image prompt editor's state — see
    "Per-image prompt overrides" above.
    """
    my_run = bshared.start_run(STAGE)

    if not files:
        yield [], "No images to process. Please drag and drop some images first."
        return
    if bshared.SLOT:
        save_to_source = save_base_copy = False

    skip_errors = shared.opts.batch_hires_fix_skip_errors

    # Snapshot the store: the running generator holds the gr.State by reference,
    # and a stray edit could otherwise mutate it mid-run. The live box is folded
    # in because its .input event may not have landed before the Run click did.
    prompts = dict(prompt_store or {})
    if sel is not None and 0 <= int(sel) < len(files) and isinstance(files[int(sel)], str):
        prompts[files[int(sel)]] = live_prompt or ""
    # Stagehand: each character's box rewrites her line in the prompt the pass uses
    for path, edits in dict(char_store or {}).items():
        own = prompts[path] if path in prompts else bshared.image_prompt(path)
        edited = bshared.apply_character_edits(own, edits)
        if edited != own:
            prompts[path] = edited

    hires_params = {
        "denoising_strength": float(denoising_strength),
        "hr_scale": float(hr_scale),
        "hr_upscaler": str(hr_upscaler) if hr_upscaler else None,
        "hr_second_pass_steps": int(hr_second_pass_steps),
        "hr_resize_x": int(hr_resize_x or 0),
        "hr_resize_y": int(hr_resize_y or 0),
        "hr_sampler_name": str(hr_sampler_name) if hr_sampler_name else None,
        "hr_scheduler": str(hr_scheduler) if hr_scheduler else None,
        "hr_cfg": float(hr_cfg),
    }

    total = len(files)
    all_results: list = []
    status_messages: list[str] = []
    failed_count = 0

    for idx, file_obj in enumerate(files):
        if bshared.cancel_requested(STAGE, my_run):
            status_messages.append(f"⏹️ Cancelled — {idx} of {total} images processed.")
            break

        if isinstance(file_obj, str):
            image_path = file_obj
        elif hasattr(file_obj, "name"):
            image_path = file_obj.name
        else:
            status_messages.append(
                f"❌ [{idx + 1}/{total}] Unknown file object at index {idx}"
            )
            failed_count += 1
            continue

        fname = os.path.basename(image_path)
        # status lines read "Commission 137 - M, Fluorite/3r1.png"
        name = bshared.display_name(image_path) if save_to_source else fname

        try:
            img = Image.open(image_path)
            # Read infotext BEFORE converting — convert() can drop PNG info.
            geninfo, _items = images.read_info_from_image(img)
            img = img.convert("RGB")
        except Exception as e:
            status_messages.append(f"❌ [{idx + 1}/{total}] Failed to load {name}: {e}")
            failed_count += 1
            continue

        # None = no override for this image: it runs with its own baked-in prompt.
        prompt_override = prompts.get(image_path)

        if not geninfo:
            status_messages.append(
                f"⚠️ [{idx + 1}/{total}] {name}: no generation info found in image — "
                + ("your edited prompt still applies, but the image's own sampler, "
                   "seed and steps are unknown."
                   if prompt_override is not None else
                   "hires pass will run with an empty prompt.")
            )

        shared.total_tqdm.clear()

        save_opts = {
            "use_original_name": bool(use_original_name),
            "stem": os.path.splitext(fname)[0],
            "suffix": filename_suffix or "",
            "base_copy": bool(save_base_copy),
        }
        if save_to_source:
            # The downstream pipeline (and the pending scan) key on
            # <stem>-hires.png, so the name and format are forced here rather
            # than taken from the suffix box: a custom suffix would leave the
            # base image "pending" forever and reprocess it every run.
            # png also keeps the infotext, which jpg/jxl/... silently drop.
            save_opts["use_original_name"] = True
            save_opts["suffix"] = "-hires"
            save_opts["format"] = "png"
            if bshared.is_dragged_temp_copy(image_path):
                # A drag-dropped file whose original resolve_dropped_paths could
                # not find: dirname() is gradio's upload cache, so saving there
                # would bury the result in a folder gradio later wipes. Leave
                # output_dir unset so it falls back to the output dir, and say so.
                status_messages.append(
                    f"⚠️ [{idx + 1}/{total}] {name}: 'save into each image's own "
                    f"folder' is on, but this image wasn't found under your scan "
                    f"roots — its result goes to the output dir. Add its folder in "
                    f"Settings → Batch Hires-Fix → scan roots."
                )
            else:
                save_opts["output_dir"] = os.path.dirname(image_path)

        # state.begin() resets state.interrupted / stopping_generation, which
        # otherwise stay True forever after a UI reload (request_restart calls
        # interrupt()) and make process_images_inner return 0 images silently.
        # Real generations get this from the UI's wrap_gradio_gpu_call wrapper.
        # queue_lock, like wrap_gradio_gpu_call: without it a txt2img/API job and
        # this image would share shared.state, each begin()/end() clobbering the other.
        if bshared.SLOT:
            # forge link: the owner's Forge runs it (checked, queued), this slot keeps it
            yield all_results, f"[{idx + 1}/{total}] {name}: waiting for the GPU / running..."
            result_images, _infotexts, error_tb, notes = _run_in_slot(
                image_path, img, geninfo, hires_params, prompt_override, filename_suffix,
                cancelled=lambda: bshared.cancel_requested(STAGE, my_run))
            stopped = False
        else:
            with call_queue.queue_lock:
                shared.state.begin(job="batch_hires_fix")
                try:
                    # GPU work must run on Forge's main thread, same as the ✨ button
                    # (txt2img.py routes through main_thread.run_and_wait_result).
                    result_images, _infotexts, error_tb, notes = main_thread.run_and_wait_result(
                        _process_single_image, img, geninfo, hires_params, save_opts,
                        prompt_override,
                    )
                finally:
                    # Read before end() and inside the lock: once released, the next
                    # job's begin() resets these flags.
                    stopped = shared.state.interrupted or shared.state.stopping_generation
                    shared.state.end()

        shared.total_tqdm.clear()

        for note in dict.fromkeys(notes or []):
            status_messages.append(f"🔧 [{idx + 1}/{total}] {name}: {note}")

        if error_tb:
            status_messages.append(f"❌ [{idx + 1}/{total}] Error on {name}:\n{error_tb}")
            failed_count += 1
            if not skip_errors:
                break
            continue

        # Cancel/Interrupt pressed during this image: report and stop the batch.
        # (Checked before the next begin(), which would reset shared.state's flags.)
        if bshared.cancel_requested(STAGE, my_run) or stopped:
            status_messages.append(
                f"⏹️ [{idx + 1}/{total}] Cancelled during {name} — stopping batch."
            )
            failed_count += 1
            break

        if not result_images:
            status_messages.append(f"⚠️ [{idx + 1}/{total}] No output for {name}")
            failed_count += 1
            continue

        all_results.extend(result_images)
        status_messages.append(f"✅ [{idx + 1}/{total}] Done: {name}")

        # Stream partial results into the gallery as each image finishes.
        yield all_results, f"Processing... {idx + 1}/{total} done.\n\n" + "\n".join(status_messages)

    status_text = (
        f"Batch complete — {len(all_results)} succeeded, "
        f"{failed_count} failed/skipped out of {total}.\n\n"
        + "\n".join(status_messages)
    )

    yield all_results, status_text

def _run_in_slot(image_path, img, geninfo, hires_params, prompt_override, suffix, cancelled):
    """A forge link slot's image: run on the owner's Forge (the /hires endpoint below), checked
    at the size it comes out, and saved in this slot. -> like _process_single_image."""
    params = bshared.parse_generation_parameters(geninfo or "", [])
    prompt = prompt_override if prompt_override is not None else params.get("Prompt", "")
    rx, ry = int(hires_params.get("hr_resize_x") or 0), int(hires_params.get("hr_resize_y") or 0)
    if rx or ry:
        width, height = rx or round(img.width * ry / img.height), ry or round(img.height * rx / img.width)
    else:
        scale = float(hires_params.get("hr_scale") or 1)
        width, height = round(img.width * scale), round(img.height * scale)
    body = {"image": bshared.file_b64(image_path), "hires_params": hires_params, "prompt_override": prompt_override}
    pics, infotexts, error, notes = bshared.run_remote("/hires", body, prompt, params.get("Negative prompt", ""),
                                                      width, height, cancelled=cancelled)
    if pics:
        outdir = (getattr(shared.opts, "batch_hires_fix_output_dir", None) or shared.opts.outdir_samples
                  or shared.opts.outdir_txt2img_samples)
        bshared.save_in_slot(pics, infotexts, outdir, os.path.splitext(os.path.basename(image_path))[0],
                             suffix or "-hires")
    return pics, infotexts, error, notes


def _hires_endpoint(body):
    """The owner's side of a slot's image: hires-fix it here, save nothing, send it back."""
    img, geninfo = bshared.image_from_b64(body["image"])
    (pics, infotexts, error, notes), stopped = bshared.run_on_main(
        "batch_hires_fix", _process_single_image, img, geninfo, body.get("hires_params") or {},
        {"discard": True}, body.get("prompt_override"))
    return {"images": [bshared.image_to_b64(im, infotexts[k] if k < len(infotexts) else None) for k, im in enumerate(pics)],
            "infotexts": infotexts, "notes": notes,
            "error": error or ("Stopped on amiiari's Forge." if stopped and not pics else None)}


def _on_app_started(_demo, app):
    if bshared.SLOT:
        return  # a slot only sends; the owner's Forge runs
    from fastapi import Body

    def hires(body: dict = Body(...)):
        return _hires_endpoint(body)

    app.add_api_route(f"{bshared.API}/hires", hires, methods=["POST"])


def batch_hires_fix_process_folders(
    folders,
    denoising_strength,
    hr_scale,
    hr_upscaler,
    hr_second_pass_steps,
    hr_resize_x,
    hr_resize_y,
    hr_sampler_name,
    hr_scheduler,
    hr_cfg,
    save_base_copy=False,
    prompt_store=None,
    char_store=None,
):
    """
    Folder mode: hires-fix every pending base image in the selected Tests
    folders, saving each result (plus its -base Lanczos twin, if ticked) back
    into the folder it came from.
    """
    if bshared.SLOT:
        yield [], "Folders aren't available here: drop your images in instead."
        return
    if not folders:
        yield [], "No folders selected — tick at least one (🔄 Rescan if the list is stale)."
        return

    # A ticked set covers its own folder AND its Tests folder.
    files = [f for folder in folders for d in bshared.set_scan_dirs(folder)
             for f in _pending_bases(d)]
    if not files:
        yield [], ("Nothing to do — every base image in the selected sets "
                   "already has a -hires (or later) version.")
        return

    # Original-name saving AND the "-hires" suffix are forced (the suffix
    # textbox is ignored here): <stem>-hires.png next to its source is what the
    # pending-scan and the content manager key on — a custom suffix would
    # leave base images "pending" forever and reprocess them every run.
    yield from batch_hires_fix_process(
        files,
        denoising_strength,
        hr_scale,
        hr_upscaler,
        hr_second_pass_steps,
        hr_resize_x,
        hr_resize_y,
        hr_sampler_name,
        hr_scheduler,
        hr_cfg,
        True,
        "-hires",
        save_base_copy=save_base_copy,
        save_to_source=True,
        # Edits made after a "📥 Load for editing" still apply if the same
        # images are run from the folder button (both key on the real path).
        prompt_store=prompt_store,
        char_store=char_store,
    )


# ──────────────────────────────────────────────
# Gradio UI Tab
# ──────────────────────────────────────────────
def _editing_label(paths, sel):
    if not paths:
        return "*No images loaded — drop images or load a folder to edit prompts.*"
    if sel is None or not (0 <= int(sel) < len(paths)):
        return "*Click a thumbnail to edit the prompt its hires pass will use.*"
    return (f"**Editing:** `{os.path.basename(paths[int(sel)])}`"
            f"  ({int(sel) + 1}/{len(paths)})")


# Every loader returns this same batch of updates:
# [source_gallery, paths_state, sel_state, editing_md, prompt_box, preview_img].
_EDITOR_OUTPUTS = 6


def _editor_updates(paths, store):
    """A freshly loaded image list, with the first image selected and the prompt
    its hires pass would use in the editor."""
    if not paths:
        return [gr.update(value=None), [], None, _editing_label([], None),
                gr.update(value=""), gr.update(value=None)]
    return [gr.update(value=paths, selected_index=0), paths, 0,
            _editing_label(paths, 0), gr.update(value=_prompt_for(store, paths[0])),
            gr.update(value=paths[0])]


def _on_files(files, suffix_filter, store):
    """Files dropped/browsed: keep only those whose stem ends with the suffix
    filter (blank = keep everything), trade gradio's temp copies for the on-disk
    originals, and load the survivors into the editor. Returns
    [*_EDITOR_OUTPUTS, file_input, status_text]."""
    paths = bshared.file_paths(files)
    paths, skipped = bshared.filter_suffix(paths, suffix_filter)

    # Rewrite the drop zone only when we actually filtered something out. That
    # rewrite retriggers this handler; the second pass skips nothing and returns
    # gr.update() with no value, which ends the loop (same guard as adetailer).
    file_update = gr.update(value=paths or None) if skipped else gr.update()

    # AFTER file_update: the run reads paths_state, not the drop zone, so the
    # originals never round-trip through gr.File (which would re-copy them into
    # gradio's cache and lose the folder again).
    paths, notes = bshared.resolve_dropped_paths(paths, STAGE)

    if not (files or []):
        status = ""
    else:
        note = (f" Skipped {skipped} not ending in `{(suffix_filter or '').strip()}`."
                if skipped else "")
        status = " ".join([f"Loaded {len(paths)} image(s)." + note, *notes])
    return [*_editor_updates(paths, store), file_update, status]


def _folder_choices():
    return bshared.folder_choices(STAGE)


def _load_folder_path(path, store):
    """📂 Load Folder: every base image in one folder, done or not, loaded
    with its real path so save-to-source lands results back in it. Bypasses the
    Test Folders panel, which is a to-do list and so can't show a finished set —
    or a set that keeps its images outside a Tests folder at all.
    Returns [*_EDITOR_OUTPUTS, save_to_source, status_text]."""
    folder = (path or "").strip().strip('"')
    files = _base_images(folder) if folder else []
    if not files:
        return [*([gr.update()] * _EDITOR_OUTPUTS), gr.update(), (
            "No base images in that folder — check the path. "
            "(This stage's inputs are plain <name>.png files; -hires / -adetailer / "
            "-edited / -base files aren't bases.)"
        )]
    return [*_editor_updates(files, store), gr.update(value=True), (
        f"Loaded {len(files)} image(s) from {folder} — results save back into it as "
        f"<name>-hires.png. 🚀 Run Batch Hires-Fix when ready."
    )]


def _load_pending_folders(folders, store):
    """📥 Load for editing: the same pending images 🚀 Hires-Fix Selected
    Folders would run, loaded into the gallery instead — so their prompts can be
    edited first. Ticks save-to-source, since these paths are the real ones.
    Returns [*_EDITOR_OUTPUTS, save_to_source, status_text]."""
    files = [f for folder in (folders or []) if folder
             for d in bshared.set_scan_dirs(folder) for f in _pending_bases(d)]
    if not files:
        return [*([gr.update()] * _EDITOR_OUTPUTS), gr.update(), (
            "Nothing to load — tick at least one set with pending base "
            "images (🔄 Rescan if the list is stale)."
        )]
    return [*_editor_updates(files, store), gr.update(value=True), (
        f"Loaded {len(files)} pending image(s) — click a thumbnail to edit the "
        f"prompt its hires pass will use, then 🚀 Run Batch Hires-Fix."
    )]


# ──────────────────────────────────────────────
# Prompt editor handlers
# ──────────────────────────────────────────────
def _on_select_image(store, paths, evt: gr.SelectData):
    """Thumbnail clicked: show that image and the prompt its hires pass will use.
    sel_state is returned in the SAME outputs batch as the box, so nothing can
    write the newly loaded prompt back into the previously selected image."""
    idx = int(evt.index)
    paths = list(paths or [])
    if not (0 <= idx < len(paths)):
        return [gr.update()] * 4
    return [idx, _editing_label(paths, idx),
            gr.update(value=_prompt_for(store, paths[idx])),
            gr.update(value=paths[idx])]


def _on_right_click(store, paths, index, txt2img_prompt=""):
    """Thumbnail right-clicked (javascript/batch_adetailer.js puts the index in a
    hidden textbox and clicks a hidden button — gradio has no contextmenu event):
    select that image and fill the box with its entire baked-in prompt, falling
    back to the live txt2img prompt (stashed by the JS) when the image has none.
    Same as the ADetailer tab's right-click. With neither, the box is left as is.
    Returns [sel_state, editing_md, prompt_box, preview_img, prompt_store]."""
    paths = list(paths or [])
    try:
        idx = int(index)
    except (TypeError, ValueError):
        idx = -1
    if not (0 <= idx < len(paths)):
        return [*([gr.update()] * 4), store]

    store = dict(store or {})
    path = paths[idx]
    prompt = bshared.image_prompt(path) or (txt2img_prompt or "").strip()
    if prompt:
        store[path] = prompt
    return [idx, _editing_label(paths, idx),
            gr.update(value=_prompt_for(store, path)),
            gr.update(value=path), store]


def _on_prompt_edit(store, paths, sel, prompt):
    """Prompt box typed in: persist it against the selected image. Bound to
    .input, not .change — a programmatic reload of the box must never be written
    back as if it were an edit."""
    if sel is None or not paths or not (0 <= int(sel) < len(paths)):
        return store
    store = dict(store or {})
    store[paths[int(sel)]] = prompt or ""
    return store


def _on_revert(store, paths, sel):
    """↺ drop this image's override and re-read its own baked-in prompt."""
    if sel is None or not paths or not (0 <= int(sel) < len(paths)):
        return store, gr.update(), "Click a thumbnail first."
    path = paths[int(sel)]
    store = dict(store or {})
    store.pop(path, None)
    return store, gr.update(value=bshared.image_prompt(path)), (
        f"↺ `{os.path.basename(path)}` reset to the prompt baked into the image."
    )


def _on_first_revision(store, paths, sel):
    """📜 fill the box with the prompt from this image's first revision."""
    if sel is None or not paths or not (0 <= int(sel) < len(paths)):
        return store, gr.update(), "Click a thumbnail first."
    if bshared.SLOT:
        return store, gr.update(), "Not available here."
    path = paths[int(sel)]
    prompt, note = _first_revision_prompt(path)
    if prompt is None:
        return store, gr.update(), f"⚠️ {os.path.basename(path)}: {note}"
    store = dict(store or {})
    store[path] = prompt
    return store, gr.update(value=prompt), f"📜 {os.path.basename(path)}: {note}"


def _export_prompts(store, paths, folder):
    """Snapshot the edited prompts (keyed by <set>/<file>) into a timestamped JSON,
    so they survive a restart."""
    if bshared.SLOT:
        return "Export isn't available here."
    if not paths:
        return "Nothing to export — load images first."
    folder = (folder or "").strip().strip('"')
    if not folder:
        return "Set an export folder first."
    data = {bshared.export_key(p): (store or {})[p] for p in paths if p in (store or {})}
    if not data:
        return "Nothing to export — no edited prompts yet."
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, time.strftime("batch_hires_fix_prompts_%Y%m%d_%H%M%S.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"💾 Exported {len(data)} prompt(s) → {out}"


def _import_prompts(file, store, paths, sel):
    """Merge an exported JSON back onto the loaded images, matched by <set>/<file>, else
    a unique filename (full paths differ between sessions — moved folders, gradio temp copies).
    Returns [store, prompt_box, status_text]."""
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if not path:
        return store, gr.update(), gr.update()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("not a Batch Hires-Fix prompt export")
    except Exception as e:
        return store, gr.update(), f"⚠️ Couldn't read that export: {e}"

    if not paths:
        return store, gr.update(), (
            "⚠️ Load your images first, then import — entries are matched by filename."
        )

    store = dict(store or {})
    matched = 0
    for p in paths:
        prompt = bshared.lookup_export(data, p)
        if isinstance(prompt, str):
            store[p] = prompt
            matched += 1

    box = gr.update()
    if sel is not None and 0 <= int(sel) < len(paths) and paths[int(sel)] in store:
        box = gr.update(value=store[paths[int(sel)]])

    msg = f"📥 Imported prompts for {matched} of {len(paths)} loaded image(s)."
    if matched < len(data):
        msg += f" ({len(data) - matched} entries in the file had no matching image.)"
    return store, box, msg


def _build_ui_tab():
    from modules import sd_samplers, sd_schedulers

    upscaler_choices = list(shared.latent_upscale_modes.keys()) + [x.name for x in shared.sd_upscalers]

    default_upscaler = "4xUltrasharp_4xUltrasharpV10"
    if default_upscaler not in upscaler_choices:
        default_upscaler = "Latent"

    with gr.Blocks(analytics_enabled=False) as block:
        gr.Markdown(
            "# Batch Hires-Fix\n"
            "Drag and drop images generated via txt2img to run them through hires-fix in batch.\n\n"
            "*Click a thumbnail on the right to edit the prompt **that image's** hires pass "
            "will use — right-click one to refill the box with its entire prompt — "
            "it starts as the prompt baked into the image. \u2190/\u2192 steps through the "
            "thumbnails (when you're not typing in a box).*"
        )

        gr.HTML(
            """
            <style>
            /* Same treatment as the ADetailer tab: drag the bottom-right corner
               to see more than one row of thumbnails. Gradio has no resizable
               gallery, but the block is just a div — `resize` is all it takes.
               The block is a flex column so the thumbnail grid *follows* the
               dragged height instead of stopping at its own max-height. */
            #batch_hires_fix_source {
                height: 200px;
                min-height: 140px;
                resize: vertical;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            #batch_hires_fix_source .grid-wrap,
            #batch_hires_fix_source .grid-container,
            #batch_hires_fix_source .gallery-container {
                flex: 1 1 auto;
                height: auto !important;
                max-height: none !important;
                min-height: 0 !important;
                overflow-y: auto;
            }
            /* The drop zone's file list grows with every image dropped. */
            #batch_hires_fix_files { max-height: 220px; overflow-y: auto; }
            /* The preview: the WHOLE image is always visible, scaled to fit the
               box, however small the box is dragged. The container is pinned to
               the box's bounds (absolute inset) so no intermediate wrapper can
               size itself to the image's natural height and crop it against
               overflow:hidden. */
            #batch_hires_fix_preview {
                height: 400px;
                min-height: 160px;
                resize: vertical;
                overflow: hidden;
                position: relative;
            }
            #batch_hires_fix_preview .image-container {
                position: absolute;
                inset: 0;
                height: auto !important;
            }
            #batch_hires_fix_preview .image-container button,
            #batch_hires_fix_preview .image-container img {
                width: 100%;
                height: 100%;
                max-height: none !important;
                object-fit: contain;
            }
            </style>
            """
        )

        # The run reads this, not the drop zone: dropped paths are gradio's temp
        # copies, and the originals must not round-trip back through gr.File.
        paths_state = gr.State([])

        # Per-image prompt overrides, {path: prompt}. An image with no entry runs
        # with the prompt read from its own infotext, as it always did.
        prompt_store = gr.State({})
        # {path: [one per Stagehand character]}: the per-character boxes' edits
        char_state = gr.State({})
        sel_state = gr.State(None)

        # Driven from javascript/batch_adetailer.js on right-click: gradio has no
        # contextmenu event, so the JS writes the thumbnail index (and the live
        # txt2img prompt, for the fallback) here and clicks the button.
        rclick_index = gr.Textbox(visible=False, elem_id="batch_hires_fix_rclick")
        rclick_prompt = gr.Textbox(visible=False, elem_id="batch_hires_fix_rclick_prompt")
        rclick_btn = gr.Button(visible=False, elem_id="batch_hires_fix_rclick_btn")

        # Test Folders panel spans the full width at the top, like the ADetailer tab.
        with gr.Accordion("📁 Test Folders — hires-fix in place", open=True, visible=not bshared.SLOT):
            folder_select = _folder_choices()
            with gr.Row():
                folder_btn = gr.Button(
                    "🚀 Hires-Fix Selected Folders", variant="primary", scale=3
                )
                # The button above goes straight from scan to GPU; this one stops
                # at the gallery so the prompts can be edited first.
                load_folders_btn = gr.Button("📥 Load for editing", scale=2)
                refresh_btn = gr.Button("🔄 Rescan", scale=1)

            # The panel above is a to-do list, so a finished set is invisible and
            # a set that keeps its images outside a Tests folder never appears at
            # all. This loads any folder as-is, done or not.
            with gr.Row():
                folder_path = gr.Textbox(
                    label="…or load every base image in one folder, done or not",
                    placeholder=r"C:\art\Commission 12 - Example",
                    max_lines=1,
                    scale=4,
                )
                load_path_btn = gr.Button("📂 Load Folder", scale=1)

        # The drop zone + filter also span the full width at the top: parked in
        # the narrow left column they crowd the settings.
        with gr.Row():
            file_input = gr.File(
                label="Drop images here (or click to browse)",
                elem_id="batch_hires_fix_files",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                scale=4,
            )
            suffix_filter = gr.Textbox(
                value="",
                label="Only load files ending with",
                info="Drag a whole folder in — anything else is skipped. "
                     "Empty = load everything.",
                max_lines=1,
                scale=1,
                visible=not bshared.SLOT,
            )
            if bshared.SLOT:
                # A slot's uploads are all meant (it can't drag folders in), and the slots'
                # ui-config.json was seeded with a stale saved filter ("-adetailer" here), which
                # would silently skip every image: no filter, and none loaded from there.
                suffix_filter.do_not_save_to_config = True

        with gr.Accordion("💾 Export / import per-image prompts", open=False, visible=not bshared.SLOT):
            with gr.Row():
                export_dir = gr.Textbox(
                    value="",
                    label="Export folder",
                    placeholder=r"C:\art\prompt exports",
                    max_lines=1,
                    scale=3,
                )
                export_btn = gr.Button("💾 Export prompts", scale=1)
                import_file = gr.File(
                    label="Import — drop an exported .json here (after loading the images)",
                    file_types=[".json"],
                    type="filepath",
                    scale=2,
                )

        with gr.Row():
            # ── Left column: the per-image prompt editor, then the settings ──
            # Same running order as the ADetailer tab: the image being edited
            # sits at the top, whole and scaled to fit — it doesn't need to be
            # big, it needs to show what's being configured. No `height`: the
            # CSS above sizes the box and gives it a drag handle.
            with gr.Column(scale=1):
                preview_img = gr.Image(
                    label="Selected image",
                    elem_id="batch_hires_fix_preview",
                    interactive=False,
                    show_download_button=False,
                )

                editing_md = gr.Markdown(_editing_label([], None))

                # The elem_id deliberately carries the real ADetailer's txt2img
                # prefix: tag autocomplete (sd-webui-tagcomplete) finds its
                # third-party targets via
                # `[id^=script_txt2img_adetailer_ad_prompt] textarea`, so
                # matching the prefix gets autocomplete in this box with no
                # tagcomplete configuration.
                prompt_box = gr.Textbox(
                    label="Prompt for the hires pass",
                    elem_id="script_txt2img_adetailer_ad_prompt_batch_hires",
                    placeholder="Click a thumbnail on the right to load its prompt.",
                    lines=8,
                    max_lines=24,
                )

                # Stagehand images: a box per character in the prompt above
                char_boxes = bshared.character_boxes("txt2img_prompt_batch_char", "prompt")

                with gr.Row():
                    revert_btn = gr.Button("↺ Revert to the image's own prompt", scale=1)
                    first_rev_btn = gr.Button("📜 Use the first revision's prompt", scale=1, visible=not bshared.SLOT)

                gr.Markdown("### Hires-Fix Settings")

                with gr.Row():
                    denoising_strength = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=0.3, label="Denoising Strength", scale=2,
                    )
                    hr_scale = gr.Slider(
                        minimum=1.0, maximum=8.0, step=0.05,
                        value=1.25, label="Upscale By", scale=2,
                    )

                hr_upscaler = gr.Dropdown(
                    choices=upscaler_choices,
                    value=default_upscaler,
                    label="Hires Upscaler",
                    allow_custom_value=True,
                )

                # hr_cfg=1.0 would make the pipeline drop the negative prompt
                # entirely, so keep this visible. The hires distilled CFG
                # (shift) is intentionally NOT exposed: it inherits each
                # image's own base shift from infotext.
                hr_cfg = gr.Slider(
                    minimum=1.0, maximum=24.0, step=0.5,
                    value=4.5, label="Hires CFG Scale",
                )

                with gr.Row():
                    hr_second_pass_steps = gr.Slider(
                        minimum=0, maximum=150, step=1,
                        value=0, label="Hires Steps (0 = same as image's steps)", scale=2,
                    )
                    hr_resize_x = gr.Number(
                        value=0, label="Resize to Width (0 = auto)",
                        min=0, precision=0, scale=2,
                    )

                hr_resize_y = gr.Number(
                    value=0, label="Resize to Height (0 = auto)",
                    min=0, precision=0,
                )

                # ── Sampler & Scheduler ──
                with gr.Row():
                    hr_sampler_name = gr.Dropdown(
                        choices=["Use same sampler"] + sd_samplers.visible_sampler_names(),
                        value="Use same sampler",
                        label="Hires sampling method",
                    )
                    hr_scheduler = gr.Dropdown(
                        choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers],
                        value="Use same scheduler",
                        label="Hires schedule type",
                    )

                # ── Output naming ──
                with gr.Row():
                    use_original_name = gr.Checkbox(
                        value=True,
                        label="Save as original filename + suffix",
                        scale=2,
                        visible=not bshared.SLOT,  # a slot always does
                    )
                    filename_suffix = gr.Textbox(
                        value="-hires",
                        label="Filename suffix",
                        max_lines=1,
                        scale=1,
                    )

                save_base_copy = gr.Checkbox(
                    value=False,
                    visible=not bshared.SLOT,
                    label="Also save a plain Lanczos upscale with no hires pass "
                          "(<name>-base.png, same size as the result)",
                )

                # Ticked automatically by "📂 Load Folder". Forces <name>-hires.png,
                # so the suffix box above is ignored while this is on.
                save_to_source = gr.Checkbox(
                    value=not bshared.SLOT,
                    label="Save as <name>-hires.png into each image's own folder",
                    visible=not bshared.SLOT,
                )

                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 Run Batch Hires-Fix", variant="primary", size="lg", scale=3
                    )
                    cancel_btn = gr.Button("⏹️ Cancel", variant="stop", size="lg", scale=1)

            # ── Right column: output & status ──
            with gr.Column(scale=2):
                # The images the run will actually process (so the suffix
                # filter's effect is visible) and the prompt editor's selector.
                # No `height`: the CSS above gives the block a starting height
                # and a drag handle, and the thumbnails scroll inside it.
                # allow_preview=False keeps a click a *selection* instead of
                # popping open the full-size viewer.
                source_gallery = gr.Gallery(
                    label="Loaded images — click one to edit the prompt its hires pass will use, "
                          "right-click to refill it with the image's entire prompt",
                    elem_id="batch_hires_fix_source",
                    columns=[4],
                    preview=False,
                    allow_preview=False,
                    show_download_button=False,
                    interactive=False,
                )

                # elem_id must end in "_gallery" so Forge's lightbox modal
                # (javascript/imageviewer.js + ui.js all_gallery_buttons) picks
                # it up — that's what enables ←/→ arrow-key navigation in the
                # full-size preview. preview=True matches the txt2img gallery.
                output_gallery = gr.Gallery(
                    label="Results",
                    elem_id="batch_hires_fix_gallery",
                    columns=[4],
                    height="auto",
                    preview=True,
                )

                status_text = gr.TextArea(
                    label="Status / Log",
                    lines=10,
                    interactive=False,
                )

        refresh_btn.click(
            fn=_folder_choices,
            inputs=[],
            outputs=[folder_select],
            queue=False,
        )

        # The character boxes follow the prompt box (the selected image's prompt, as edited):
        # refreshed after everything that rewrites it, and as it's typed in.
        def chars_refresh(chars, paths, sel, prompt):
            path = paths[int(sel)] if paths and sel is not None and 0 <= int(sel) < len(paths) else None
            return bshared.character_box_updates(prompt or "", (chars or {}).get(path), "prompt")

        def chars_expand(chars, paths, sel, prompt):
            """Right-click: every character's box written out from the prompt box."""
            if not paths or sel is None or not 0 <= int(sel) < len(paths) or not bshared.image_characters(prompt):
                return chars
            return {**(chars or {}), paths[int(sel)]: bshared.expanded_characters(prompt)}

        def chars_edit(chars, paths, sel, *boxes):
            if not paths or sel is None or not 0 <= int(sel) < len(paths):
                return chars
            return {**(chars or {}), paths[int(sel)]: list(boxes)}

        refresh = dict(fn=chars_refresh, inputs=[char_state, paths_state, sel_state, prompt_box], outputs=char_boxes,
                       queue=False, show_progress="hidden")

        # Every loader writes the same batch: the gallery, the paths the run
        # reads, the selection, and the editor showing the first image.
        editor_outputs = [source_gallery, paths_state, sel_state, editing_md,
                          prompt_box, preview_img]

        load_path_btn.click(
            fn=_load_folder_path,
            inputs=[folder_path, prompt_store],
            outputs=[*editor_outputs, save_to_source, status_text],
            queue=False,
        ).then(**refresh)

        load_folders_btn.click(
            fn=_load_pending_folders,
            inputs=[folder_select, prompt_store],
            outputs=[*editor_outputs, save_to_source, status_text],
            queue=False,
        ).then(**refresh)

        folder_btn.click(
            fn=batch_hires_fix_process_folders,
            inputs=[
                folder_select,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_sampler_name,
                hr_scheduler,
                hr_cfg,
                save_base_copy,
                prompt_store,
                char_state,
            ],
            outputs=[output_gallery, status_text],
        ).then(  # the run consumed pending work — rescan so the list stays honest
            fn=_folder_choices,
            inputs=[],
            outputs=[folder_select],
        )

        # queue=False so the click is served straight away instead of queueing
        # behind the running batch — otherwise the cancel could never arrive.
        cancel_btn.click(
            fn=lambda: bshared.request_cancel(STAGE),
            inputs=[],
            outputs=[status_text],
            queue=False,
        )

        # Filter + preview dropped files. queue=False keeps it snappy and stops it
        # queueing behind a running batch. Set the suffix filter BEFORE dropping —
        # like adetailer, it only re-filters when the file list itself changes.
        file_input.change(
            fn=_on_files,
            inputs=[file_input, suffix_filter, prompt_store],
            outputs=[*editor_outputs, file_input, status_text],
            queue=False,
        ).then(**refresh)

        # ── prompt editor ──
        source_gallery.select(
            fn=_on_select_image,
            inputs=[prompt_store, paths_state],
            outputs=[sel_state, editing_md, prompt_box, preview_img],
            queue=False,
        ).then(**refresh)

        rclick_btn.click(
            fn=_on_right_click,
            inputs=[prompt_store, paths_state, rclick_index, rclick_prompt],
            outputs=[sel_state, editing_md, prompt_box, preview_img, prompt_store],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=chars_expand, inputs=[char_state, paths_state, sel_state, prompt_box], outputs=char_state,
            queue=False, show_progress="hidden",
        ).then(**refresh)

        # .input, not .change: only a keystroke is an edit. A .change would also
        # fire when a *selection* rewrites the box, writing the newly loaded
        # prompt back into whichever image was selected a moment ago.
        prompt_box.input(
            fn=_on_prompt_edit,
            inputs=[prompt_store, paths_state, sel_state, prompt_box],
            outputs=[prompt_store],
            queue=False,
            show_progress="hidden",
        ).then(**refresh)  # a character line added or removed shows or hides her box

        # .input: a keystroke is an edit; a refresh setting the boxes is not
        gr.on(
            triggers=[b.input for b in char_boxes],
            fn=chars_edit,
            inputs=[char_state, paths_state, sel_state, *char_boxes],
            outputs=char_state,
            queue=False,
            show_progress="hidden",
        )

        revert_btn.click(
            fn=_on_revert,
            inputs=[prompt_store, paths_state, sel_state],
            outputs=[prompt_store, prompt_box, status_text],
            queue=False,
        ).then(**refresh)

        first_rev_btn.click(
            fn=_on_first_revision,
            inputs=[prompt_store, paths_state, sel_state],
            outputs=[prompt_store, prompt_box, status_text],
            queue=False,
        ).then(**refresh)

        export_btn.click(
            fn=_export_prompts,
            inputs=[prompt_store, paths_state, export_dir],
            outputs=[status_text],
            queue=False,
        )

        import_file.change(
            fn=_import_prompts,
            inputs=[import_file, prompt_store, paths_state, sel_state],
            outputs=[prompt_store, prompt_box, status_text],
            queue=False,
        )

        process_btn.click(
            fn=batch_hires_fix_process,
            inputs=[
                paths_state,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_sampler_name,
                hr_scheduler,
                hr_cfg,
                use_original_name,
                filename_suffix,
                save_base_copy,
                save_to_source,
                prompt_store,
                sel_state,
                prompt_box,
                char_state,
            ],
            outputs=[output_gallery, status_text],
        ).then(  # a save-to-source run consumed pending work — keep the list honest
            fn=_folder_choices,
            inputs=[],
            outputs=[folder_select],
        )

    return block

def _on_ui_tabs():
    """Register the Batch Hires-Fix tab with Forge Neo's UI."""
    yield (_build_ui_tab(), "Batch Hires-Fix", "batch-hires-fix-tab")

# ──────────────────────────────────────────────
# Registration
# Runs unconditionally: Forge Neo loads each extension script once per launch.
# Do NOT guard on shared.opts attribute existence — saved values in config.json
# make the attribute exist before registration, which would skip tab
# registration entirely.
# ──────────────────────────────────────────────
_register_settings()
script_callbacks.on_ui_tabs(_on_ui_tabs)
script_callbacks.on_app_started(_on_app_started)
