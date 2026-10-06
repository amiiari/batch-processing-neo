"""
Batch ADetailer Extension for Forge Neo
=======================================
A UI tab where you drag in multiple images and run each through ADetailer
(detect + inpaint faces/hands/...) in batch — with per-image settings.

Each dropped image carries its own configuration: click a thumbnail on the right,
and that image's unit slots load on the left. Per slot you pick which of your
ADetailer units to use (which brings that unit's model and all its saved settings
along) and override just the things that vary per image: the ADetailer prompt and
negative prompt, detection confidence, inpaint denoising strength, and mask max
area ratio. Slot order is execution order, so a hand unit can be run before a face
unit on an image where the hand overlaps the face.

The processing trick: ADetailer is an alwayson script whose real work happens in
postprocess_image(). Its "skip img2img" flag neuters the base img2img pass (1 step,
Euler, 128x128 — nearly free) and makes it detect+inpaint on p.init_images[0] at
full resolution instead. So a batch run is just: build one
StableDiffusionProcessingImg2Img per image with init_images=[img], hand it
ADetailer's args with skip_img2img=True, and let process_images() do the rest.
"""
import copy as _copy
import importlib
import json
import os
import re
import sys
import time
import traceback
from contextlib import closing
from functools import partial

import gradio as gr
from PIL import Image

from modules import call_queue, images, processing, script_callbacks, scripts, shared
from modules_forge import main_thread

import batch_adetailer_shared as bshared
# scripts/*.py are re-executed on every in-process Reload UI, but root modules
# stay cached in sys.modules — reload so shared-code edits land too.
importlib.reload(bshared)

STAGE = bshared.ADETAILER_STAGE

ADETAILER_TITLE = "adetailer"

# The per-image overrides, in the order their controls appear in each slot.
OVERRIDE_ATTRS = [
    "ad_prompt",
    "ad_negative_prompt",
    "ad_confidence",
    "ad_denoising_strength",
    "ad_mask_max_ratio",
]
# preset dropdown + one control per override
CONTROLS_PER_SLOT = 1 + len(OVERRIDE_ATTRS)

PRESET_DISABLED = -1

# _BASE_PROMPT_RE rewrites the old "[base prompt]" spelling to ADetailer's own
# "[PROMPT]" placeholder (adetailer/scripts/!adetailer.py :: _get_prompt) for
# backward compat with prompts saved before v0.5.
_BASE_PROMPT_RE = re.compile(r"\[\s*base\s*prompt\s*\]", re.IGNORECASE)

DEFAULT_NUM_SLOTS = 4

# Safety net if ADetailer's pydantic model can't be reached to enumerate fields.
_FALLBACK_AD_FIELDS = {
    "ad_model", "ad_model_classes", "ad_tab_enable", "ad_hires_fix_only",
    "ad_prompt", "ad_negative_prompt", "ad_confidence", "ad_mask_filter_method",
    "ad_mask_k", "ad_mask_min_ratio", "ad_mask_max_ratio", "ad_dilate_erode",
    "ad_x_offset", "ad_y_offset", "ad_mask_merge_invert", "ad_mask_blur",
    "ad_denoising_strength", "ad_inpaint_only_masked",
    "ad_inpaint_only_masked_padding", "ad_use_inpaint_width_height",
    "ad_inpaint_width", "ad_inpaint_height", "ad_inpaint_scale", "ad_use_steps",
    "ad_steps", "ad_use_cfg_scale", "ad_cfg_scale", "ad_use_checkpoint",
    "ad_checkpoint", "ad_use_vae", "ad_vae", "ad_use_sampler", "ad_sampler",
    "ad_scheduler", "ad_use_noise_multiplier", "ad_noise_multiplier",
    "ad_use_clip_skip", "ad_clip_skip", "ad_restore_face", "ad_controlnet_model",
    "ad_controlnet_module", "ad_controlnet_weight", "ad_controlnet_guidance_start",
    "ad_controlnet_guidance_end",
}

# ──────────────────────────────────────────────
# Extension Settings (registered via add_option)
# ──────────────────────────────────────────────
def _register_settings():
    section = ("batch_adetailer", "Batch ADetailer")

    shared.opts.add_option(
        "batch_adetailer_output_dir",
        shared.OptionInfo("", "Output Directory", gr.Textbox, {}, section=section)
        .info("Leave empty to use the default img2img output directory."),
    )

    shared.opts.add_option(
        "batch_adetailer_scan_roots",
        shared.OptionInfo(
            "",
            "Test-folder scan roots (semicolon-separated)", gr.Textbox, {}, section=section)
        .info("Each root is searched (up to 3 levels deep) for Tests folders — so one "
              "root covers both <root>/<set>/Tests and <root>/Commissions/<set>/Tests. "
              "The same roots are used to find drag-dropped images back on disk. "
              "Non-existent roots are silently skipped."),
    )

    shared.opts.add_option(
        "batch_adetailer_skip_errors",
        shared.OptionInfo(True, "Skip Failed Images and Continue", gr.Checkbox,
                          {}, section=section)
        .info("If an image fails during ADetailer, skip it and continue with the rest."),
    )

    shared.opts.add_option(
        "batch_adetailer_fix_lora_names",
        shared.OptionInfo(True, "Repair Unresolvable LoRA Names in Prompts", gr.Checkbox,
                          {}, section=section)
        .info(
            "A prompt read from an old image can name a LoRA that no longer exists under "
            "that name (a training epoch like 'mylora-000021' that was later renamed to "
            "'mylora'). Forge can't resolve it and renders without the LoRA. When on, such "
            "names are re-pointed at the matching file in your Lora folder. Covers both "
            "the Batch ADetailer and Batch Hires-Fix tabs."
        ),
    )

# ──────────────────────────────────────────────
# Test-folder scanning — shared core, parameterized by this stage's record.
#
# ADetailer runs SECOND in the refine chain, on the hires-fix results:
#   NrM.png -> NrM-hires.png -> NrM-hires-adetailer.png
# A -hires image is "pending" while it has no -adetailer (or -edited) successor.
# Compositional edits are new revisions (1r2), never suffixes.
# ──────────────────────────────────────────────
def _hires_images(folder):
    return bshared.stage_inputs(folder, STAGE)


def _pending_hires(folder):
    return bshared.pending_inputs(folder, STAGE)


# ──────────────────────────────────────────────
# Talking to the installed ADetailer extension
# ──────────────────────────────────────────────
def _find_adetailer_script():
    """
    The installed ADetailer alwayson script object on the img2img runner, or
    None if the extension isn't installed/enabled for img2img. We need the
    object itself (not just its presence) for its args_from/args_to slice and
    for the gradio components it created.
    """
    try:
        return scripts.scripts_img2img.script(ADETAILER_TITLE)
    except Exception:
        return None


def _get_num_slots():
    """
    How many ADetailer unit slots exist. Comes from ADetailer's `ad_max_models`
    setting, which decides how many gr.State unit slots its script_args slice has
    (2 leading bools + one slot per unit).
    """
    ad_script = _find_adetailer_script()
    if ad_script is None:
        return DEFAULT_NUM_SLOTS
    try:
        slots = (ad_script.args_to - ad_script.args_from) - 2
        return slots if slots >= 1 else DEFAULT_NUM_SLOTS
    except Exception:
        return DEFAULT_NUM_SLOTS


def _get_adetailer_defaults():
    """
    The user's *saved* img2img ADetailer defaults, one dict per unit — i.e. what
    they configured via Settings → Defaults (ui-config.json), not ADetailer's
    stock values.

    ADetailer builds one `gr.State(lambda: state_init(w))` per unit, where
    state_init reads `{attr: widget.value}` off its live widgets at call time.
    Gradio keeps that lambda in `component.load_event_to_attach`. Meanwhile
    Forge's UiLoadsave *mutates* those widgets' .value with the saved defaults
    (ui_loadsave.py: `setattr(obj, field, saved_value)`), but only after all
    script UI is built. So calling the lambda now re-reads the widgets and yields
    the user's defaults; `state.value` alone would only give the build-time
    (stock) values.

    MUST be called at event/click time — at UI-build time UiLoadsave hasn't run
    yet and this returns stock defaults.
    """
    ad_script = _find_adetailer_script()
    if ad_script is None or not getattr(ad_script, "controls", None):
        return []

    out = []
    for state in ad_script.controls[2:]:  # [ad_enable, ad_skip_img2img, *unit states]
        values = None

        load_event = getattr(state, "load_event_to_attach", None)
        if load_event:
            try:
                values = dict(load_event[0]())
            except Exception:
                values = None

        if not isinstance(values, dict) or not values:
            values = dict(getattr(state, "value", None) or {})

        values.pop("is_api", None)  # leave ADetailerArgs' own default in place
        out.append(values)

    return out


def _ad_field_names():
    """
    Valid ADetailerArgs field names. The model is `extra=Extra.forbid`, so a
    single stray key makes the whole unit fail validation and get dropped
    silently — everything we hand over gets filtered through this.
    """
    try:
        model = sys.modules["adetailer"].ADetailerArgs
        fields = set(getattr(model, "__fields__", None) or model.model_fields)
        if fields:
            return fields
    except Exception:
        pass
    return set(_FALLBACK_AD_FIELDS)


def _preset_choices(defaults):
    """
    Dropdown choices for "which of my ADetailer units does this slot use", as
    (label, value) pairs — the label shows the unit's model so the user can tell
    the face unit from the hand unit at a glance.
    """
    choices = [("— slot disabled —", PRESET_DISABLED)]
    for i, unit in enumerate(defaults):
        model = str(unit.get("ad_model", "None") or "None")
        label = f"Unit {i + 1} — {model}" if model != "None" else f"Unit {i + 1} — (no model set)"
        choices.append((label, i))
    return choices

# ──────────────────────────────────────────────
# Per-image config
#
# One image's config is a flat list of NUM_SLOTS * CONTROLS_PER_SLOT values,
# laid out slot by slot as [preset, prompt, negative, confidence, denoise,
# max_ratio] — matching the order of the controls built by _slot_controls().
# ──────────────────────────────────────────────
def _blank_slot_values():
    return [PRESET_DISABLED, "", "", 0.3, 0.4, 1.0]


def _slot_values_from_unit(preset_index, unit):
    return [
        preset_index,
        str(unit.get("ad_prompt", "") or ""),
        str(unit.get("ad_negative_prompt", "") or ""),
        float(unit.get("ad_confidence", 0.3)),
        float(unit.get("ad_denoising_strength", 0.5)),
        float(unit.get("ad_mask_max_ratio", 1.0)),
    ]


def _default_config(defaults, num_slots):
    """
    A fresh image's config: slot i uses unit i (same as ADetailer's own layout),
    with the overrides pre-filled from that unit's saved defaults.
    """
    config = []
    for i in range(num_slots):
        unit = defaults[i] if i < len(defaults) else None
        # A unit with no model set can't detect anything, so leave that slot
        # disabled rather than pre-selecting a unit that would never run.
        if unit and str(unit.get("ad_model", "None") or "None") != "None":
            config += _slot_values_from_unit(i, unit)
        else:
            config += _blank_slot_values()
    return config


def _config_to_unit_dicts(config, defaults, num_slots):
    """
    One image's config -> the list of ADetailer unit dicts to run, in slot order
    (which is execution order). Each dict is that preset's full saved defaults
    with the per-image overrides applied on top.
    """
    units = []

    for i in range(num_slots):
        chunk = config[i * CONTROLS_PER_SLOT : (i + 1) * CONTROLS_PER_SLOT]
        if len(chunk) < CONTROLS_PER_SLOT:
            continue

        preset = chunk[0]
        if preset is None or int(preset) < 0 or int(preset) >= len(defaults):
            continue

        unit = dict(defaults[int(preset)])
        if not unit:
            continue

        unit["ad_prompt"] = str(chunk[1] or "")
        unit["ad_negative_prompt"] = str(chunk[2] or "")
        unit["ad_confidence"] = float(chunk[3])
        unit["ad_denoising_strength"] = float(chunk[4])
        unit["ad_mask_max_ratio"] = float(chunk[5])

        # Without these, a unit can silently no-op: a preset whose tab default is
        # off would need_skip(), and hires-fix-only never matches our img2img pass.
        unit["ad_tab_enable"] = True
        unit["ad_hires_fix_only"] = False

        if str(unit.get("ad_model", "None") or "None") == "None":
            continue

        allowed = _ad_field_names()
        units.append({k: v for k, v in unit.items() if k in allowed})

    return units

def _assemble_script_args(unit_dicts):
    """
    Full flat script_args array for the img2img runner, with ADetailer's slice
    replaced by [ad_enable, ad_skip_img2img, unit0, unit1, ...].

    Every unit slot must be overwritten: ADetailer's own UI defaults put gr.State
    *lambdas* in those positions, and while ADetailer ignores non-dict args, a
    leftover lambda would silently mean "one fewer unit than the user configured".
    Unused slots get an inert {"ad_model": "None"} (ADetailer's need_skip()).

    Returns (script_args, warning_or_None), or (None, error) if ADetailer is absent.
    """
    ad_script = _find_adetailer_script()
    if ad_script is None:
        return None, (
            "ADetailer not found on the img2img tab. Install/enable the ADetailer "
            "extension (aadetailer-neoforge) and reload the UI."
        )

    script_args = bshared.get_default_script_args(scripts.scripts_img2img, "img2img").copy()

    slice_len = ad_script.args_to - ad_script.args_from
    num_slots = slice_len - 2  # minus the two leading bools
    if num_slots < 1:
        return None, (
            f"Unexpected ADetailer args layout (slice length {slice_len}). "
            "Is the installed ADetailer version compatible?"
        )

    warning = None
    active = list(unit_dicts)
    if len(active) > num_slots:
        warning = (
            f"Only {num_slots} ADetailer unit slot(s) available — units beyond "
            f"#{num_slots} were ignored. Raise 'Max models' in Settings → ADetailer "
            f"and restart to use all {len(active)}."
        )
        active = active[:num_slots]

    ad_slice = [True, True]  # ad_enable, ad_skip_img2img
    ad_slice += active
    ad_slice += [{"ad_model": "None"}] * (num_slots - len(active))

    script_args[ad_script.args_from : ad_script.args_to] = ad_slice
    return script_args, warning

# ──────────────────────────────────────────────
# Saving with the original filename + suffix
# ──────────────────────────────────────────────
def _fix_infotext(infotext: str | None, width: int, height: int, steps: int | None,
                  sampler: str | None = None):
    """
    Undo the cosmetic damage skip-img2img does to the saved infotext: the base
    pass is neutered to 1 step, Euler, 128x128 *before* create_infotext runs,
    so the result would advertise those for a full-res image. Downstream tools
    (batch-hires-fix) inherit per-image params from this infotext, so it must
    tell the truth.
    """
    if not infotext:
        return infotext

    infotext = re.sub(r"(?<=\bSize: )128x128\b", f"{width}x{height}", infotext)
    if steps:
        infotext = re.sub(r"(?<=\bSteps: )1(?=,|$)", str(steps), infotext)
    if sampler and sampler != "Euler":
        infotext = re.sub(r"(?<=\bSampler: )Euler(?=,|$)", sampler, infotext)
    return infotext


_SEP_RE = re.compile(r"\s*\[SEP\]\s*")


def _collapse_prompt_editing(unit_dicts, p):
    """
    ADetailer re-samples each face from step 0, so `[from:to:7]` in the source
    prompt would inpaint `from` again for 7 steps. Resolve the blank/[PROMPT]
    fallback to the state the finished image was left in. Explicitly authored
    ADetailer schedules are left for its inpaint pass to run normally.

    Collapse only the fallback, before inserting it into each [SEP] segment:
    final_prompt parks LoRA tags at the end of its input, so passing the joined
    face prompts would move every face's LoRA into the last segment. The image's
    own Prompt metadata is untouched. p.styles are already folded into p.prompt.
    """
    fallbacks = {"ad_prompt": p.prompt, "ad_negative_prompt": p.negative_prompt}
    for unit in unit_dicts:
        for key, fallback in fallbacks.items():
            parts = _SEP_RE.split(unit.get(key, "") or "")
            if not any(not part or "[PROMPT]" in part for part in parts):
                continue
            final = bshared.final_prompt(fallback, p.steps)
            if final != fallback:  # no source schedules -> let ADetailer resolve it
                unit[key] = "[SEP]".join(
                    final if not part else part.replace("[PROMPT]", final)
                    for part in parts
                )


# ──────────────────────────────────────────────
# Core Processing Logic
# ──────────────────────────────────────────────
def _process_single_image(img: Image.Image, geninfo: str | None, unit_dicts: list, save_opts: dict):
    """
    Run one image through ADetailer.

    Runs on Forge's main thread (see batch_adetailer_process). Never raises:
    returns (images, infotexts, error_traceback_or_None, notes) so the full
    traceback reaches the status log instead of being swallowed by Gradio.
    """
    notes: list[str] = []
    try:
        unit_dicts = [dict(u) for u in unit_dicts]
        for unit in unit_dicts:
            for key in ("ad_prompt", "ad_negative_prompt"):
                # "[base prompt]" -> ADetailer's own [PROMPT] placeholder, which it
                # substitutes with the image's prompt (p.all_prompts) at inpaint time.
                text = _BASE_PROMPT_RE.sub("[PROMPT]", unit.get(key, "") or "")
                unit[key], found = bshared.fix_lora_names(text)
                notes += found

        p = processing.StableDiffusionProcessingImg2Img(
            outpath_samples=(
                save_opts.get("output_dir")
                or getattr(shared.opts, "batch_adetailer_output_dir", None)
                or shared.opts.outdir_samples
                or shared.opts.outdir_img2img_samples
            ),
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
            prompt="",
            negative_prompt="",
            styles=[],
            batch_size=1,
            n_iter=1,
            cfg_scale=7.0,
            init_images=[img],
            width=img.size[0],
            height=img.size[1],
            resize_mode=0,
            # The base img2img pass is neutered by ADetailer's skip-img2img
            # (1 step / 128x128), so this value never actually shapes the output —
            # but it must still be a legal denoising strength.
            denoising_strength=0.4,
            # mask must stay None: ADetailer disables skip-img2img on inpaint
            # processing objects (it calls that combination buggy).
            mask=None,
            mask_blur=4,
            inpainting_fill=1,
            inpaint_full_res=False,
            inpaint_full_res_padding=32,
            inpainting_mask_invert=0,
            override_settings={},
        )

        # Scripts are assigned HERE, before the image's own parameters are
        # applied: steps, sampler, scheduler and seed are alwayson scripts in
        # Forge (modules/processing_scripts/sampler.py :: setup, seed.py ::
        # setup), and the scripts/script_args setters run setup_scripts() the
        # moment both are set (processing.py :: script_args.setter). Assigning
        # them *after* apply_source_image_parameters lets those setups overwrite
        # every inherited value with the live img2img UI's (20 steps, DPM++ 2M,
        # Normal, random seed) — which silently neuters the inpaint.
        # The real ADetailer args are swapped in further down, once the prompts
        # they carry are final.
        p.scripts = scripts.scripts_img2img
        p.script_args = bshared.get_default_script_args(
            scripts.scripts_img2img, "img2img").copy()

        if geninfo:
            # ADetailer's inpaint pass inherits these: an empty ad_prompt falls
            # back to p.prompt, and with skip-img2img the steps/sampler it uses
            # come from p (captured into p._ad_orig before the base pass is
            # neutered).
            params = bshared.apply_source_image_parameters(p, geninfo)
            # Fold the recovered styles in now: ADetailer's blank/[PROMPT] fallback
            # reads the already-styled p.all_prompts and then builds its inpaint
            # with styles=p.styles, which would apply every style a second time.
            p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
            p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(
                p.negative_prompt, p.styles)
            p.styles = []

        # A slot with a blank ADetailer prompt inpaints with *this* prompt (ADetailer
        # falls back to p.all_prompts), so a LoRA that can't be resolved here is a
        # LoRA missing from the inpaint.
        p.prompt, found = bshared.fix_lora_names(p.prompt)
        notes += found
        p.negative_prompt, found = bshared.fix_lora_names(p.negative_prompt)
        notes += found

        # An image made with Stagehand's Character Prompts carries its characters as lines
        # in the prompt. Inlined into the units here, every face would get every character;
        # Stagehand gives each face its own character and resolves the schedules itself.
        if not bshared.has_characters(p.prompt):
            _collapse_prompt_editing(unit_dicts, p)

        script_args, _warning = _assemble_script_args(unit_dicts)
        if script_args is None:
            return [], [], _warning, notes
        if geninfo:
            # Scripts that can rebuild their settings from the image's own get them, so its
            # metadata carries them on to the hires tab (Stagehand's Precise Reference).
            bshared.replay_script_args(scripts.scripts_img2img, script_args, params)

        # Plain reassignment: setup_scripts() already ran above (guarded by
        # scripts_setup_complete), so this only swaps ADetailer's slice in — it
        # can't re-run the core setups and clobber the inherited parameters.
        # ADetailer itself has no setup(), so it loses nothing by being late.
        p.script_args = script_args

        print(f"[Batch ADetailer] base prompt: {p.prompt!r}")
        print(f"[Batch ADetailer] steps={p.steps} sampler={p.sampler_name!r} "
              f"scheduler={getattr(p, 'scheduler', None)!r} seed={p.seed}")
        if p.styles:
            print(f"[Batch ADetailer] styles: {p.styles}")
        for note in dict.fromkeys(notes):
            print(f"[Batch ADetailer] {note}")

        # Captured before process_images(), because ADetailer's process() hook
        # overwrites p.steps/width/height/sampler with its 1-step/Euler/128x128
        # stand-ins.
        orig_size = (p.width, p.height)
        orig_steps = p.steps
        orig_sampler = p.sampler_name

        # Always saved by hand afterwards: core's own save would write the
        # skip-img2img infotext (Steps: 1, Euler, 128x128), and the hires tab
        # then inherits a 1-step pass from it.
        p.do_not_save_samples = True

        with closing(p):
            processed = scripts.scripts_img2img.run(p, *p.script_args)

            if processed is None:
                processed = processing.process_images(p)

        if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
            # An interrupted inpaint returns a partial/unfinished result;
            # saving it would make the image look finished (and with
            # save-to-source would hide it from the pending scan forever) —
            # drop it instead.
            return [], [], None, notes

        fix_info = lambda t: _fix_infotext(t, orig_size[0], orig_size[1], orig_steps, orig_sampler)
        if save_opts.get("use_original_name"):
            bshared.save_with_original_name(processed, p, save_opts, fix_info=fix_info)
        else:
            for i, image in enumerate(processed.images):
                info = processed.infotexts[i] if i < len(processed.infotexts) else None
                images.save_image(image, p.outpath_samples, "", p.all_seeds[0], p.all_prompts[0],
                                  shared.opts.samples_format, info=fix_info(info), p=p)

        return processed.images, processed.infotexts, None, notes
    except Exception:
        tb = traceback.format_exc()
        print(f"[Batch ADetailer] Error processing image:\n{tb}")
        return [], [], tb, notes


def batch_adetailer_run_selected(store, paths, sel, use_original_name, filename_suffix, save_to_source, *control_values):
    """
    Re-run just the selected image — for when a batch came out fine except for one
    or two. It's the batch loop over a single path, so the config, saving and
    cancelling all behave identically. The result doesn't overwrite the earlier
    one: save_with_original_name adds a -1, -2, ... counter on collision.
    """
    paths = list(paths or [])
    if sel is None or not (0 <= int(sel) < len(paths)):
        yield "Click a thumbnail first — this button runs the image you have selected."
        return

    yield from batch_adetailer_process(
        store, [paths[int(sel)]], 0, use_original_name, filename_suffix, save_to_source, *control_values
    )


def batch_adetailer_process(store, paths, sel, use_original_name, filename_suffix, save_to_source, *control_values):
    """
    Main batch processing function. Each image is processed with its own config
    from the store, sequentially.

    save_to_source: each result is saved next to its own source image as
    <stem>-adetailer.png (suffix and png forced — that naming is what the
    pending scan and the rest of the pipeline key on).

    control_values are the live values of the currently-visible slot controls. We
    fold them back into the store first, so an edit that hasn't landed as a
    .change event yet still counts — the same defence ADetailer uses in its own
    on_generate_click.
    """
    my_run = bshared.start_run(STAGE)

    num_slots = _get_num_slots()

    if not paths:
        yield "No images to process. Please drag and drop some images first."
        return

    if _find_adetailer_script() is None:
        yield (
            "❌ ADetailer not found on the img2img tab.\n\n"
            "This extension drives the ADetailer extension (aadetailer-neoforge) — "
            "install/enable it and reload the UI."
        )
        return

    skip_errors = shared.opts.batch_adetailer_skip_errors

    # Snapshot the store: the running generator holds the gr.State by reference,
    # and a stray .change event could otherwise mutate it mid-run.
    store = _copy.deepcopy(dict(store or {}))
    if sel is not None and 0 <= int(sel) < len(paths):
        store[paths[int(sel)]] = list(control_values)

    defaults = _get_adetailer_defaults()
    if not defaults:
        yield (
            "❌ Could not read your ADetailer unit defaults from the img2img panel.\n\n"
            "Open the img2img tab once, then come back and try again."
        )
        return

    total = len(paths)
    all_results: list = []
    status_messages: list[str] = []
    failed_count = 0

    for idx, image_path in enumerate(paths):
        fname = os.path.basename(image_path)
        # Every set has a 1r1.png — prefix the set name so the status lines read
        # "Commission 137 - M, Fluorite/1r1.png" and stay tellable apart.
        name = bshared.display_name(image_path) if save_to_source else fname

        if bshared.cancel_requested(STAGE, my_run):
            status_messages.append(f"⏹️ Cancelled — {idx} of {total} images processed.")
            break

        config = store.get(image_path)
        if not config:
            config = _default_config(defaults, num_slots)

        try:
            unit_dicts = _config_to_unit_dicts(config, defaults, num_slots)
        except (TypeError, ValueError) as e:
            # A malformed import (unvalidated JSON) mustn't kill the whole batch.
            status_messages.append(
                f"❌ [{idx + 1}/{total}] {name}: bad per-image settings ({e}) — skipped."
            )
            failed_count += 1
            continue
        if not unit_dicts:
            status_messages.append(
                f"⚠️ [{idx + 1}/{total}] {name}: no enabled unit slots — skipped."
            )
            failed_count += 1
            continue

        try:
            img = Image.open(image_path)
            # Read infotext BEFORE converting — convert() can drop PNG info.
            geninfo, _items = images.read_info_from_image(img)
            img = img.convert("RGB")
        except Exception as e:
            status_messages.append(f"❌ [{idx + 1}/{total}] Failed to load {name}: {e}")
            failed_count += 1
            continue

        if not geninfo:
            status_messages.append(
                f"⚠️ [{idx + 1}/{total}] {name}: no generation info found in image — "
                f"slots with an empty ADetailer prompt will inpaint with no prompt."
            )

        shared.total_tqdm.clear()

        save_opts = {
            "use_original_name": bool(use_original_name),
            "stem": os.path.splitext(fname)[0],
            "suffix": filename_suffix or "",
        }
        if save_to_source:
            save_opts["use_original_name"] = True
            save_opts["suffix"] = "-adetailer"
            # png keeps the infotext and is what the pipeline expects, even if
            # the global samples_format is jpg/jxl/...
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
                    f"Settings → Batch ADetailer → scan roots."
                )
            else:
                save_opts["output_dir"] = os.path.dirname(image_path)

        # state.begin() resets state.interrupted / stopping_generation, which
        # otherwise stay True forever after a UI reload (request_restart calls
        # interrupt()) and make process_images_inner return 0 images silently.
        # Real generations get this from the UI's wrap_gradio_gpu_call wrapper.
        # queue_lock, like wrap_gradio_gpu_call: without it a txt2img/API job and
        # this image would share shared.state, each begin()/end() clobbering the other.
        with call_queue.queue_lock:
            shared.state.begin(job="batch_adetailer")
            try:
                # GPU work must run on Forge's main thread, same as img2img()
                # (img2img.py routes through main_thread.run_and_wait_result).
                result_images, _infotexts, error_tb, notes = main_thread.run_and_wait_result(
                    _process_single_image, img, geninfo, unit_dicts, save_opts
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
        models = ", ".join(str(u.get("ad_model")) for u in unit_dicts)
        status_messages.append(f"✅ [{idx + 1}/{total}] Done: {name}  ({models})")

        # Stream progress into the log as each image finishes.
        yield f"Processing... {idx + 1}/{total} done.\n\n" + "\n".join(status_messages)

    yield (
        f"Batch complete — {len(all_results)} succeeded, "
        f"{failed_count} failed/skipped out of {total}.\n\n"
        + "\n".join(status_messages)
    )

# ──────────────────────────────────────────────
# UI event handlers
# ──────────────────────────────────────────────
def _editing_label(paths, sel):
    if not paths:
        return "### No images loaded\nDrop images above to configure them."
    if sel is None or not (0 <= int(sel) < len(paths)):
        return "### Click a thumbnail to edit that image's units"
    return (
        f"### Editing: `{os.path.basename(paths[int(sel)])}`"
        f"  ({int(sel) + 1}/{len(paths)})"
    )


def _control_updates(config, num_slots, choices=None):
    """gr.updates for every slot control, from one image's config."""
    updates = []
    for i in range(num_slots):
        chunk = config[i * CONTROLS_PER_SLOT : (i + 1) * CONTROLS_PER_SLOT]
        if len(chunk) < CONTROLS_PER_SLOT:
            chunk = _blank_slot_values()
        if choices is not None:
            updates.append(gr.update(choices=choices, value=chunk[0]))
        else:
            updates.append(gr.update(value=chunk[0]))
        updates += [gr.update(value=v) for v in chunk[1:]]
    return updates


def _load_paths(paths, store, num_slots, skip_note=""):
    """
    Load a list of image paths into the tab: seed a config for each new image
    from the user's saved ADetailer defaults, keep configs for images that were
    already loaded, and show the first image's config. Returns updates for
    [gallery, paths_state, store_state, sel_state, editing_md, *controls,
    preview].
    """
    store = dict(store or {})
    defaults = _get_adetailer_defaults()
    choices = _preset_choices(defaults)

    if not paths:
        blank = _default_config(defaults, num_slots)
        return [
            gr.update(value=None),        # source gallery
            [],                           # paths_state
            {},                           # store_state
            None,                         # sel_state
            _editing_label([], None) + skip_note,
            *_control_updates(blank, num_slots, choices),
            gr.update(value=None),        # preview
        ]

    store = {
        path: store.get(path) or _default_config(defaults, num_slots)
        for path in paths
    }

    return [
        gr.update(value=paths, selected_index=0),
        paths,
        store,
        0,
        _editing_label(paths, 0) + skip_note,
        *_control_updates(store[paths[0]], num_slots, choices),
        gr.update(value=paths[0]),
    ]


def _on_files(files, store, num_slots, suffix_filter):
    """
    Files dropped: keep only files whose stem ends with the suffix filter (so a
    whole folder can be dragged in and only the `-hires` variants load), then
    load them. The paths gradio gives us are temp-cache copies, so they're traded
    back for the on-disk originals first — otherwise "save into each image's own
    folder" would target gradio's cache. (Folder mode still does NOT route through
    this box: its paths are already the real ones.)
    """
    paths = bshared.file_paths(files)
    paths, skipped = bshared.filter_suffix(paths, suffix_filter)

    # Push the filtered list back into the drop zone so it matches what loaded.
    # gr.update() (no value) when nothing was skipped — this runs in the drop
    # zone's own .change handler, so an unconditional write would retrigger it
    # forever; the retrigger after a filtering pass filters nothing and stops.
    file_update = gr.update(value=paths or None) if skipped else gr.update()
    skip_note = (
        f"\n\n*Skipped {skipped} file(s) not ending in `{(suffix_filter or '').strip()}`.*"
        if skipped else ""
    )

    paths, notes = bshared.resolve_dropped_paths(paths, STAGE)
    for note in notes:
        skip_note += f"\n\n*{note}*"

    out = _load_paths(paths, store, num_slots, skip_note)
    out.insert(len(out) - 1, file_update)  # the outputs list puts the drop zone before the preview
    return out


def _on_select_image(store, paths, num_slots, evt: gr.SelectData):
    """
    Thumbnail clicked: load that image's config. sel_state is returned in the
    SAME outputs batch as the controls — if it were updated in a later event, the
    change-storm from setting the controls would write the newly loaded values
    back into the previously selected image's slot and corrupt it.
    """
    idx = int(evt.index)
    paths = list(paths or [])
    if not (0 <= idx < len(paths)):
        return [gr.update()] * (3 + num_slots * CONTROLS_PER_SLOT)

    config = (store or {}).get(paths[idx]) or _default_config(
        _get_adetailer_defaults(), num_slots
    )

    return [
        idx, _editing_label(paths, idx),
        *_control_updates(config, num_slots),
        gr.update(value=paths[idx]),
    ]


def _on_right_click(store, paths, index, num_slots, img2img_prompt=""):
    """
    Thumbnail right-clicked (via javascript/batch_adetailer.js, which puts the
    index in a hidden textbox and clicks a hidden button): select that image and
    fill Slot 1's ADetailer prompt with that image's entire prompt, read from its
    embedded generation info. If the image has no embedded prompt, fall back to
    the live img2img prompt (stashed by the JS). If neither has anything, the slot is left untouched. Everything else
    about the slot is left as is.
    """
    paths = list(paths or [])
    try:
        idx = int(index)
    except (TypeError, ValueError):
        idx = -1

    blank = [gr.update()] * (3 + num_slots * CONTROLS_PER_SLOT)
    if not (0 <= idx < len(paths)):
        return [*blank, store]

    store = dict(store or {})
    config = list(
        store.get(paths[idx]) or _default_config(_get_adetailer_defaults(), num_slots)
    )
    prompt = bshared.image_prompt(paths[idx])
    if not prompt:
        # No prompt baked into the image — fall back to the live img2img prompt.
        prompt = (img2img_prompt or "").strip()
    if bshared.has_characters(prompt):
        # A Stagehand image (or a live prompt with character lines): written out here, every
        # face would get every character (and the labels). [PROMPT] lets Stagehand give each
        # face its own character's prompt.
        prompt = "[PROMPT]"
    if prompt:
        config[1] = prompt  # slot 1's prompt — position 0 is its preset dropdown
    store[paths[idx]] = config

    return [
        idx, _editing_label(paths, idx),
        *_control_updates(config, num_slots),
        gr.update(value=paths[idx]),
        store,
    ]


def _on_control_change(store, paths, sel, *control_values):
    """Any slot control edited: persist it into the selected image's config."""
    if sel is None or not paths or not (0 <= int(sel) < len(paths)):
        return store
    store = dict(store or {})
    store[paths[int(sel)]] = list(control_values)
    return store


def _on_preset_change(store, paths, sel, preset, slot, num_slots):
    """
    Slot's unit preset changed: repopulate that slot's overrides from the chosen
    unit's saved defaults, and persist. Bound separately from the generic control
    handler so the two never race on the same trigger.
    """
    defaults = _get_adetailer_defaults()

    preset_idx = int(preset) if preset is not None else PRESET_DISABLED
    if 0 <= preset_idx < len(defaults):
        values = _slot_values_from_unit(preset_idx, defaults[preset_idx])
    else:
        values = _blank_slot_values()
        values[0] = preset_idx

    store = dict(store or {})
    if sel is not None and paths and 0 <= int(sel) < len(paths):
        path = paths[int(sel)]
        config = list(store.get(path) or _default_config(defaults, num_slots))
        # .change also fires when selecting/importing an image sets the dropdown to
        # that image's stored preset; refilling then would wipe its saved prompts.
        if config[slot * CONTROLS_PER_SLOT] == preset_idx:
            return [store, *[gr.update() for _ in values[1:]]]
        config[slot * CONTROLS_PER_SLOT : (slot + 1) * CONTROLS_PER_SLOT] = values
        store[path] = config

    return [store, *[gr.update(value=v) for v in values[1:]]]


def _on_apply_to_all(store, paths, sel, *control_values):
    """
    Copy the currently-shown config onto every loaded image — except the prompts.
    Those are the per-image part (that's the whole point of the tab), so each image
    keeps its own; everything else (unit choice, confidence, denoise, max ratio)
    is overwritten.
    """
    if not paths:
        return store, "No images loaded."

    source = list(control_values)
    num_slots = len(source) // CONTROLS_PER_SLOT
    store = dict(store or {})
    sel_path = paths[int(sel)] if sel is not None and 0 <= int(sel) < len(paths) else None

    for path in paths:
        config = list(source)
        # The selected image's prompts are the live control values themselves, so
        # only the *other* images have prompts of their own to preserve.
        if path != sel_path:
            existing = list(store.get(path) or source)
            for i in range(num_slots):
                prompt_at = i * CONTROLS_PER_SLOT + 1  # [preset, prompt, negative, ...]
                if prompt_at + 1 < len(existing):
                    config[prompt_at] = existing[prompt_at]
                    config[prompt_at + 1] = existing[prompt_at + 1]
        store[path] = config

    src = os.path.basename(paths[int(sel)]) if sel is not None and 0 <= int(sel) < len(paths) else "current"
    return store, (
        f"Applied `{src}` settings to all {len(paths)} images "
        f"(each image kept its own prompts)."
    )

def _export_prompts(store, paths, folder):
    """Snapshot every loaded image's slot configs (keyed by <set>/<file>) into a
    timestamped JSON file, so the per-image prompts survive a restart."""
    if not paths:
        return "Nothing to export — load images first."
    folder = (folder or "").strip().strip('"')
    if not folder:
        return "Set an export folder first."
    data = {bshared.export_key(p): (store or {}).get(p) for p in paths}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return "Nothing to export — no per-image settings yet."
    out = os.path.join(folder, time.strftime("batch_adetailer_prompts_%Y%m%d_%H%M%S.json"))
    os.makedirs(folder, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"💾 Exported settings for {len(data)} image(s) → {out}"


def _import_prompts(file, store, paths, sel, num_slots):
    """Merge an exported JSON back onto the loaded images, matched by <set>/<file>, else
    a unique filename (full paths differ between sessions — gradio temp copies, moved folders).
    Returns [store, *control updates, status]."""
    noop = [gr.update()] * (num_slots * CONTROLS_PER_SLOT)
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if not path:
        return [store, *noop, gr.update()]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("not a Batch ADetailer export")
    except Exception as e:
        return [store, *noop, f"⚠️ Couldn't read that export: {e}"]

    if not paths:
        return [store, *noop,
                "⚠️ Load your images first, then import — entries are matched by filename."]

    want = num_slots * CONTROLS_PER_SLOT
    store = dict(store or {})
    matched = 0
    for p in paths:
        cfg = bshared.lookup_export(data, p)
        if not isinstance(cfg, list):
            continue
        base = list(store.get(p) or _default_config(_get_adetailer_defaults(), num_slots))
        base[: min(len(cfg), want)] = cfg[:want]
        store[p] = base
        matched += 1

    updates = noop
    if sel is not None and 0 <= int(sel) < len(paths):
        cfg = store.get(paths[int(sel)])
        if cfg:
            updates = _control_updates(cfg, num_slots)

    msg = f"📥 Imported settings for {matched} of {len(paths)} loaded image(s)."
    if matched < len(data):
        msg += f" ({len(data) - matched} entries in the file had no matching image.)"
    return [store, *updates, msg]

# ──────────────────────────────────────────────
# Gradio UI Tab
# ──────────────────────────────────────────────
def _folder_choices():
    return bshared.folder_choices(STAGE)


def _slot_controls(slot_index, num_slots):
    """
    One execution slot's controls. Order must match the config layout:
    [preset, prompt, negative, confidence, denoise, max_ratio].
    """
    # Real labels (with each unit's model) are filled in on file drop, once
    # ADetailer's saved defaults are readable.
    choices = [("— slot disabled —", PRESET_DISABLED)] + [
        (f"Unit {i + 1}", i) for i in range(num_slots)
    ]

    preset = gr.Dropdown(
        choices=choices,
        value=slot_index if slot_index < num_slots else PRESET_DISABLED,
        label="ADetailer unit (brings its model + your saved settings)",
    )

    # lines = rows shown at rest; the box grows with the text up to max_lines.
    #
    # The elem_ids deliberately carry the real ADetailer's img2img prefixes:
    # tag autocomplete (sd-webui-tagcomplete) finds its third-party targets via
    # `[id^=script_img2img_adetailer_ad_prompt] textarea` (and the negative
    # twin), so matching the prefix gets autocomplete in these boxes with no
    # tagcomplete configuration. The _batch_slot suffix keeps them unique.
    prompt = gr.Textbox(
        label="ADetailer prompt",
        elem_id=f"script_img2img_adetailer_ad_prompt_batch_slot{slot_index + 1}",
        placeholder=(
            "Empty = reuse this image's own prompt from its metadata. "
            "Or write [PROMPT] to build on it: '[PROMPT], detailed eyes'"
        ),
        lines=5,
        max_lines=20,
    )
    negative_prompt = gr.Textbox(
        label="ADetailer negative prompt",
        elem_id=f"script_img2img_adetailer_ad_negative_prompt_batch_slot{slot_index + 1}",
        placeholder="Empty = reuse this image's own negative prompt ([PROMPT] works here too)",
        lines=5,
        max_lines=20,
    )

    with gr.Row():
        confidence = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.01,
            value=0.3, label="Detection confidence",
        )
        denoising_strength = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.01,
            value=0.5, label="Inpaint denoising strength",
        )

    mask_max_ratio = gr.Slider(
        minimum=0.0, maximum=1.0, step=0.001,
        value=1.0, label="Mask max area ratio (ignore detections bigger than this)",
    )

    return [preset, prompt, negative_prompt, confidence, denoising_strength, mask_max_ratio]


def _build_ui_tab():
    num_slots = _get_num_slots()

    with gr.Blocks(analytics_enabled=False) as block:
        gr.Markdown(
            "# Batch ADetailer\n"
            "Drop images, click a thumbnail to configure **that image's** units, then run the batch. "
            "Each slot pulls its model and settings from one of your ADetailer units "
            "(as saved in the img2img panel) — you only override what varies per image. "
            "Slot order is the order the units run in.\n\n"
            "*Right-click a thumbnail to fill Slot 1's prompt with "
            "that image's entire prompt (read from its metadata), or the img2img prompt if "
            "the image has none. ←/→ steps through the thumbnails (when you're not typing "
            "in a box).*"
        )

        gr.HTML(
            """
            <style>
            /* Drag the gallery's bottom-right corner to see more than one row of
               thumbnails. Gradio has no resizable gallery, but the block is just a
               div — `resize` is all it takes.
               The block is a flex column so the thumbnail grid *follows* the dragged
               height instead of stopping at its own max-height and leaving the block
               to scroll: the grid is the only thing that scrolls, and it fills
               whatever height the drag gives it. */
            #batch_adetailer_source {
                height: 200px;
                min-height: 140px;
                resize: vertical;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            #batch_adetailer_source .grid-wrap,
            #batch_adetailer_source .grid-container,
            #batch_adetailer_source .gallery-container {
                flex: 1 1 auto;
                height: auto !important;
                max-height: none !important;
                min-height: 0 !important;
                overflow-y: auto;
            }
            /* The drop zone's file list grows with every image dropped. */
            #batch_adetailer_files { max-height: 220px; overflow-y: auto; }
            /* The preview: the WHOLE image is always visible, scaled to fit
               the box, however small the box is dragged. The container is
               pinned to the box's bounds (absolute inset) so no intermediate
               wrapper can size itself to the image's natural height and crop
               it against overflow:hidden. */
            #batch_adetailer_preview {
                height: 400px;
                min-height: 160px;
                resize: vertical;
                overflow: hidden;
                position: relative;
            }
            #batch_adetailer_preview .image-container {
                position: absolute;
                inset: 0;
                height: auto !important;
            }
            #batch_adetailer_preview .image-container button,
            #batch_adetailer_preview .image-container img {
                width: 100%;
                height: 100%;
                max-height: none !important;
                object-fit: contain;
            }
            </style>
            """
        )

        paths_state = gr.State([])
        store_state = gr.State({})
        sel_state = gr.State(None)

        # Driven from javascript/batch_adetailer.js on right-click: gradio has no
        # contextmenu event, so the JS writes the thumbnail index here and clicks
        # the button.
        rclick_index = gr.Textbox(visible=False, elem_id="batch_adetailer_rclick")
        rclick_prompt = gr.Textbox(visible=False, elem_id="batch_adetailer_rclick_prompt")
        rclick_btn = gr.Button(visible=False, elem_id="batch_adetailer_rclick_btn")

        with gr.Accordion("📁 Test Folders — load pending -hires images", open=True):
            folder_select = _folder_choices()
            with gr.Row():
                load_btn = gr.Button("📥 Load Selected Folders", variant="primary", scale=3)
                rescan_btn = gr.Button("🔄 Rescan", scale=1)

            # The panel above is a to-do list, so a finished set is invisible and
            # a set that keeps its images outside a Tests folder never appears at
            # all. This loads any folder as-is, done or not.
            with gr.Row():
                folder_path = gr.Textbox(
                    label="…or load every -hires image in one folder, done or not",
                    placeholder=r"C:\art\Commission 12 - Example",
                    max_lines=1,
                    scale=4,
                )
                load_path_btn = gr.Button("📂 Load Folder", scale=1)

        # The drop zone spans the full width at the top: parked in the left column it
        # grows with the file list and pushes the unit controls off the screen.
        with gr.Row():
            file_input = gr.File(
                label="Drop images here (or click to browse)",
                elem_id="batch_adetailer_files",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                scale=4,
            )
            suffix_filter = gr.Textbox(
                value="-hires",
                label="Only load files ending with",
                info="Drag a whole folder's worth in — anything else is skipped. Empty = load everything.",
                max_lines=1,
                scale=1,
            )

        with gr.Accordion("💾 Export / import per-image prompts", open=False):
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
            # ── Left column: per-image unit editor ──
            with gr.Column(scale=1):
                # The image being edited, whole and scaled to fit — it doesn't
                # need to be big, it needs to show what's being configured.
                # No `height`: the CSS above sizes the box and gives it a
                # drag handle.
                preview_img = gr.Image(
                    label="Selected image",
                    elem_id="batch_adetailer_preview",
                    interactive=False,
                    show_download_button=False,
                )

                editing_md = gr.Markdown(_editing_label([], None))

                slot_controls: list = []
                with gr.Tabs():
                    for i in range(num_slots):
                        with gr.Tab(f"Slot {i + 1}"):
                            slot_controls.append(_slot_controls(i, num_slots))

                controls = [c for slot in slot_controls for c in slot]
                presets = [slot[0] for slot in slot_controls]
                overrides = [c for slot in slot_controls for c in slot[1:]]

                apply_all_btn = gr.Button(
                    "📋 Apply these settings to all images (keeps each image's prompts)"
                )

                with gr.Row():
                    use_original_name = gr.Checkbox(
                        value=True,
                        label="Save as original filename + suffix",
                        scale=2,
                    )
                    filename_suffix = gr.Textbox(
                        value="-adetailer",
                        label="Filename suffix",
                        max_lines=1,
                        scale=1,
                    )

                # Ticked automatically by "Load Selected Folders".
                save_to_source = gr.Checkbox(
                    value=False,
                    label="Save as <name>-adetailer.png into each image's own folder",
                )

                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 Run Batch ADetailer", variant="primary", size="lg", scale=3
                    )
                    run_one_btn = gr.Button(
                        "▶️ Run this image", variant="secondary", size="lg", scale=2
                    )
                    cancel_btn = gr.Button("⏹️ Cancel", variant="stop", size="lg", scale=1)

            # ── Right column: the thumbnail strip and the log ──
            with gr.Column(scale=2):
                # No `height`: the CSS below gives the block a starting height and a
                # drag handle, and the thumbnails scroll inside it. A gradio `height`
                # would pin the inner grid and fight the resize.
                #
                # allow_preview=False keeps a click on a thumbnail a *selection*
                # instead of popping open the full-size viewer.
                source_gallery = gr.Gallery(
                    label="Images — click or ←/→ to step through, right-click to fill Slot 1 with the image's prompt",
                    # Deliberately NOT suffixed "_gallery": that suffix is what makes
                    # Forge's imageviewer.js attach its lightbox, which we don't want
                    # on the source thumbnails.
                    elem_id="batch_adetailer_source",
                    columns=[4],
                    allow_preview=False,
                    show_download_button=False,
                    interactive=False,
                )

                status_text = gr.TextArea(
                    label="Status / Log",
                    lines=10,
                    interactive=False,
                )

        # ── wiring ──
        # These are closures rather than functools.partial: binding a keyword arg
        # with partial turns the trailing `evt: gr.SelectData` parameter into a
        # keyword-only one, and gradio only scans *positional* params when it
        # decides where to inject the event data — so the click event would never
        # be passed. Closures keep the signatures clean.
        def on_files(files, store, suffix):
            return _on_files(files, store, num_slots, suffix)

        def on_select_image(store, paths, evt: gr.SelectData):
            return _on_select_image(store, paths, num_slots, evt)

        def on_right_click(store, paths, index, img2img_prompt):
            return _on_right_click(store, paths, index, num_slots, img2img_prompt)

        file_input.change(
            fn=on_files,
            inputs=[file_input, store_state, suffix_filter],
            outputs=[source_gallery, paths_state, store_state, sel_state, editing_md,
                     *controls, file_input, preview_img],
            queue=False,
        )

        source_gallery.select(
            fn=on_select_image,
            inputs=[store_state, paths_state],
            outputs=[sel_state, editing_md, *controls, preview_img],
            queue=False,
        )

        rclick_btn.click(
            fn=on_right_click,
            inputs=[store_state, paths_state, rclick_index, rclick_prompt],
            outputs=[sel_state, editing_md, *controls, preview_img, store_state],
            queue=False,
            show_progress="hidden",
        )

        # One dependency for every override control, rather than N separate ones.
        # The preset dropdowns are deliberately NOT in here — they have their own
        # handler below, which also rewrites the overrides.
        gr.on(
            triggers=[c.change for c in overrides],
            fn=_on_control_change,
            inputs=[store_state, paths_state, sel_state, *controls],
            outputs=store_state,
            queue=False,
        )

        for i, preset in enumerate(presets):
            preset.change(
                fn=partial(_on_preset_change, slot=i, num_slots=num_slots),
                inputs=[store_state, paths_state, sel_state, preset],
                outputs=[store_state, *slot_controls[i][1:]],
                queue=False,
                show_progress="hidden",
            )

        apply_all_btn.click(
            fn=_on_apply_to_all,
            inputs=[store_state, paths_state, sel_state, *controls],
            outputs=[store_state, status_text],
            queue=False,
        )

        export_btn.click(
            fn=_export_prompts,
            inputs=[store_state, paths_state, export_dir],
            outputs=[status_text],
            queue=False,
        )

        import_file.change(
            fn=lambda file, store, paths, sel: _import_prompts(
                file, store, paths, sel, num_slots
            ),
            inputs=[import_file, store_state, paths_state, sel_state],
            outputs=[store_state, *controls, status_text],
            queue=False,
        )

        def _load_folders(folders, store, pending_only=True, empty_msg=None):
            """-hires images of the given folders, loaded directly — NOT through the
            drop zone: gradio copies every value that round-trips a gr.File into
            its temp cache, and save-to-source derives the output folder from each
            path, so the paths must stay the originals for results to land back in
            the source folders. Also ticks save-to-source."""
            pick = _pending_hires if pending_only else _hires_images
            folders = [f for f in (folders or []) if f]
            # A ticked set covers its own folder AND its Tests folder. The 📂
            # path box stays literal: pending_only=False loads exactly one dir.
            dirs = ([d for f in folders for d in bshared.set_scan_dirs(f)]
                    if pending_only else folders)
            files = [f for d in dirs for f in pick(d)]
            if not files:
                noop = [gr.update()] * (6 + num_slots * CONTROLS_PER_SLOT)
                return [*noop, gr.update(), empty_msg or (
                    "No pending -hires images — tick at least one set "
                    "(🔄 Rescan if the list is stale)."
                )]
            return [*_load_paths(files, store, num_slots), gr.update(value=True), (
                f"Loaded {len(files)} image(s) from {len(folders)} folder(s) — "
                "configure prompts, then 🚀 Run Batch ADetailer."
            )]

        folder_outputs = [source_gallery, paths_state, store_state, sel_state,
                          editing_md, *controls, preview_img, save_to_source,
                          status_text]

        load_btn.click(
            fn=_load_folders,
            inputs=[folder_select, store_state],
            outputs=folder_outputs,
            queue=False,
        )

        load_path_btn.click(
            fn=lambda path, store: _load_folders(
                [(path or "").strip().strip('"')], store, pending_only=False,
                empty_msg="No -hires images in that folder — check the path. "
                          "(This stage's inputs are <name>-hires.png files.)",
            ),
            inputs=[folder_path, store_state],
            outputs=folder_outputs,
            queue=False,
        )

        rescan_btn.click(
            fn=_folder_choices,
            inputs=[],
            outputs=[folder_select],
            queue=False,
        )

        # queue=False so the click is served straight away instead of queueing
        # behind the running batch — otherwise the cancel could never arrive.
        cancel_btn.click(
            fn=lambda: bshared.request_cancel(STAGE),
            inputs=[],
            outputs=[status_text],
            queue=False,
        )

        run_inputs = [
            store_state, paths_state, sel_state,
            use_original_name, filename_suffix, save_to_source,
            *controls,
        ]

        process_btn.click(
            fn=batch_adetailer_process,
            inputs=run_inputs,
            outputs=[status_text],
        ).then(  # a save-to-source run consumed pending work — keep the list honest
            fn=_folder_choices,
            inputs=[],
            outputs=[folder_select],
        )

        run_one_btn.click(
            fn=batch_adetailer_run_selected,
            inputs=run_inputs,
            outputs=[status_text],
        ).then(fn=_folder_choices, inputs=[], outputs=[folder_select])

    return block


def _on_ui_tabs():
    """Register the Batch ADetailer tab with Forge Neo's UI."""
    yield (_build_ui_tab(), "Batch ADetailer", "batch-adetailer-tab")

# ──────────────────────────────────────────────
# Registration
# Runs unconditionally: Forge Neo loads each extension script once per launch.
# Do NOT guard on shared.opts attribute existence — saved values in config.json
# make the attribute exist before registration, which would skip tab
# registration entirely.
# ──────────────────────────────────────────────
_register_settings()
script_callbacks.on_ui_tabs(_on_ui_tabs)
