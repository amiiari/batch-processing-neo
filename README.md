# Batch Processing for Forge Neo (Batch ADetailer + Batch Hires-Fix)

Two batch-processing tabs for [Stable Diffusion WebUI Forge — Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo),
built around one shared core. Drop in (or auto-discover) a folder's worth of
images and refine them one after another:

- **Batch ADetailer** — runs each image through ADetailer (detect + inpaint
  faces/hands/...) with **per-image** unit settings and prompts.
- **Batch Hires-Fix** — runs each image through Forge's own hires-fix pipeline,
  inheriting every image's generation parameters from its metadata, with a
  **per-image** prompt you can edit before the pass.

![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## The pipeline

The tabs are the two stages of one refine chain over work-in-progress folders:

```
1r1.png  →  1r1-hires.png  →  1r1-hires-adetailer.png
 (base)    (Batch Hires-Fix)       (Batch ADetailer)
```

Images are named `<image>r<revision>` (`1r1`, `1r2`, `10r13`); compositional
edits are new revisions, never suffixes, and only the **latest revision** of
each image number is picked up. Hires-fix runs first, then ADetailer repaints
the faces on the hires result, so the faces are the last thing touched. A
layered edit stage (e.g. the content manager's Krita stage) puts the
`-hires-adetailer` result on top of the `-hires` image, so a face repaint that
went wrong can be erased per-region. The hires stage can also save a plain
Lanczos `-base` upscale with no hires pass (off by default).

You don't have to adopt any of this to use the tabs — plain drag-and-drop with
a custom filename suffix works on any images.

## Requirements

The [ADetailer](https://github.com/Bing-su/adetailer) extension must be
installed and enabled for the **Batch ADetailer** tab (built against the
`aadetailer-neoforge` fork) — that tab drives ADetailer's own pipeline rather
than reimplementing detection or inpainting. The **Batch Hires-Fix** tab works
without it. No extra pip dependencies.

## Installation

1. Clone this repository into your Forge Neo `extensions` folder:
   ```
   cd <your Forge Neo folder>/extensions
   git clone https://github.com/amiiari/batch-processing-neo
   ```
2. Restart Forge Neo (or Reload UI).
3. Two new tabs appear: **Batch ADetailer** and **Batch Hires-Fix**.

> Upgrading from the separate `batch-hires-fix-neo` extension? Delete its
> folder — the tab now ships from here, with the same settings keys, so your
> saved settings carry over.

## Batch ADetailer

Drop your images. Click a thumbnail on the right, and that image's **slots**
load on the left. Each slot picks one of your ADetailer units — which brings
along that unit's detection model **and every setting you saved for it** in the
img2img ADetailer panel (mask blur, dilate/erode, padding, steps, CFG,
sampler, ...). You only override the handful of things that vary per image:

- **ADetailer prompt / negative prompt** (empty = reuse that image's own prompt
  from its metadata; `[PROMPT]` — or the friendlier `[base prompt]` — stands
  for that prompt with room to add to it: `[PROMPT], detailed eyes`).
  Prompt editing inherited from the source is collapsed to how the image
  finished: `[a:b:7]` becomes `b`, `[b:7]` becomes `b`, `[a::7]` becomes
  nothing (if the image finished after step 7). Schedules typed into the
  ADetailer prompt run normally, as in regular ADetailer. `[SEP]` separates
  prompts for successive detections, with each segment keeping its own LoRAs.
- **Detection confidence**
- **Inpaint denoising strength**
- **Mask max area ratio**

**Slot order is execution order.** On an image where a hand overlaps a face,
put the hand unit in Slot 1 and the face unit in Slot 2, and the face pass runs
last — over the top of the hand.

**Right-click a thumbnail** to fill Slot 1's prompt with that image's entire
prompt (read from its metadata), or the live img2img prompt if
the image has none — a quick starting point to edit from.

**▶️ Run this image** re-runs only the selected thumbnail — for when a batch
came out fine except for one or two. It saves alongside the earlier result
(`mypic-adetailer-1.png`) rather than overwriting it.

**📋 Apply these settings to all images** copies the unit choice, confidence,
denoising strength and mask max ratio onto every image — but **not the
prompts**, since those are the part that's meant to differ per image.

More on this tab:

- **No base-image regeneration** — uses ADetailer's "skip img2img" path, so the
  base pass is a throwaway 1-step 128×128 render and only the detected regions
  are actually inpainted, at full resolution. The saved metadata is repaired to
  keep the source's real steps/size/sampler, so the hires stage inherits true
  parameters.
- **Inherits your ADetailer defaults** from the img2img panel (Settings →
  Defaults), so the batch tab stays uncluttered and always matches your setup.
  The number of slots is capped by **Settings → ADetailer → Max models**.
- **Export / import per-image prompts** — snapshot every loaded image's slot
  configs to a JSON file and merge them back later (matched by filename), so
  per-image work survives a restart.
- **Tag autocomplete** — if you use sd-webui-tagcomplete, it works in the slot
  prompt boxes out of the box.
- **←/→ arrow keys** step through the thumbnails (when you're not typing in a
  box); the thumbnail gallery and preview have drag handles to resize.

## Batch Hires-Fix

Drag in images generated by txt2img (images without embedded metadata still
process, but with an empty prompt) and set the hires-fix parameters:

| Control | Meaning |
|---|---|
| Denoising Strength | how much the upscale pass re-details the image |
| Upscale By | scale factor |
| Hires Upscaler | upscaler model (`Latent` = latent-space upscale) |
| Hires CFG Scale | CFG for the hires pass (1.0 disables the negative prompt!) |
| Hires Steps | steps for the hires pass; `0` = same as the image's own step count |
| Resize to Width/Height | exact target size, `0` = use scale factor |
| Hires sampling method / schedule type | override sampler for the hires pass |
| Save as original filename + suffix | keep your file names, e.g. `pic.png → pic-hires.png` |
| Prompt for the hires pass | the selected image's prompt — see per-image prompt editing below |

- **Faithful to the ✨ button** — uses Forge's own hires-fix pipeline
  (`firstpass_image` + `process_images`), not a reimplementation.
- **Per-image prompt editing** — click a thumbnail and the box below shows the
  prompt *that image's* hires pass will use, pre-filled from its own metadata.
  Edit it and only that image changes. Because the tab feeds Forge a
  `firstpass_image`, the first pass is skipped entirely and this one prompt is
  what conditions the hires pass. Untouched images run exactly as before.
  - **Right-click a thumbnail** to select it and refill the box with the
    image's entire prompt (or the live txt2img prompt if it has none).
  - **↺ Revert** drops the edit and re-reads the image's own prompt.
  - **📜 Use the first revision's prompt** reads the prompt out of the same
    image's earliest revision on disk — select `20r3-adetailer` and it pulls
    from `20r1` (only the latest revision is ever loaded, so the earlier ones
    are otherwise out of reach).
  - **💾 Export / import** writes the edited prompts to a timestamped JSON and
    merges one back by filename, so they survive a restart.
  - **📥 Load for editing** in the Test Folders panel loads the same pending
    images 🚀 Hires-Fix Selected Folders would run, so folder mode gets the
    editor too instead of going straight to the GPU.
- **Per-image parameter inheritance** — each image's prompt, negative prompt,
  styles, seed (with variation seed and strength), steps, sampler, scheduler,
  CFG, clip skip, and shift (distilled CFG) are read from its embedded
  generation info, so every image is upscaled exactly the way it was generated.
- **`-base` Lanczos twin** — optionally saves a plain upscale of the source at
  the result's exact size (no model pass). Off by default, idempotent on
  re-runs.
- **Full lightbox preview** — click a result for the full-size viewer with ←/→
  arrow-key navigation, same as the txt2img gallery.

## Shared between both tabs

- **Test-folder mode** (below) with per-stage pending detection.
- **Keep your filenames** — results save as `<original name><suffix>.png` flat
  in the output folder, no dated subfolders or numbered naming; collisions get
  a `-1`, `-2`, ... counter instead of overwriting.
- **Drag-drop still saves to source** — the browser only uploads bytes, so a
  dropped file arrives as a temp copy with its folder lost. Dropped files are
  traded back for the on-disk originals (same name, same size, byte-identical)
  found under the scan roots, so *save into each image's own folder* works for
  drag-drop too.
- **Suffix filter on the drop zone** — drag a whole folder's worth in and only
  files ending in the suffix load (empty = load everything).
- **LoRA name repair** — an old image's prompt often names a LoRA that has
  since been renamed (a training epoch like `mylora-000021` that's now just
  `mylora`). Forge can't resolve it and quietly renders without the LoRA; both
  tabs re-point the name at the real file, and say so in the log when they
  can't.
- **Live progress** — the log fills in as each image finishes; the Cancel
  button aborts the image being worked on and stops that batch (and only that
  batch).
- **Readable errors** — failures show the full traceback in the status log and
  skip to the next image (configurable per tab).
- **Stagehand images** ([forge-stagehand](../forge-stagehand)) re-run with their characters
  and references: Character Prompts writes each character as a `Character N:` line in the
  prompt, which both tabs pass through as the prompt, and scripts that can restore their own
  settings from an image's metadata (`args_from_infotext`, used by Precise Reference) get them
  instead of their UI defaults (`replay_script_args`). Batch ADetailer leaves the prompt-editing
  collapse to Stagehand for those images, so each face gets only its own character.
- **A prompt box per character** (Stagehand images): under the slots on Batch ADetailer (her
  face prompt) and under the prompt on Batch Hires-Fix (her prompt in the hires pass), one
  for each character in the selected image, named like `Character 2 (Ruby)`. Each starts as
  `[PROMPT]`, her own prompt from the image; `[PROMPT], crying` adds to it, and anything else
  replaces it. Right-click the thumbnail to write every character's prompt out to edit. The
  edit rewrites her `Character N` line, so the result's PNG info shows what was used. The
  boxes need forge-stagehand installed (they use its reader for those lines); they aren't in
  the prompt export.

### Test-folder mode

The **📁 Test Folders** panel at the top of each tab searches the roots
configured in Settings, up to 3 levels deep, for **sets** — a set is any folder
that has a `Tests` subfolder. So one root covers sets kept directly under it
(`<root>/Commission 12`) *and* sets kept in a group folder
(`<root>/Commissions/Commission 12`). **Requests groups** are the exception:
folders inside a `Requests` folder are sets even without a `Tests` subfolder
(they keep their images directly in the folder).

One entry covers both halves of a set: its own folder and its `Tests` folder
are scanned and loaded together. Nothing else is — archives under
`Tests/Finished`, a `Reference` folder and loose scratch directories are not
work in progress.

The panel is a **to-do list** of pending work:

- On **Batch Hires-Fix**, a set is listed while it has base images (like
  `3r1.png`) with no `-hires` version next to them. Bases that already have a
  plain `-adetailer` sibling went through the old adetailer-first chain and are
  left alone.
- On **Batch ADetailer**, a set is listed while it has `-hires` images with no
  `-adetailer` or `-edited` successor.

Tick the sets you want and load/run them: every result saves back **next to its
own source image** with the stage's suffix (`<name>-hires.png` /
`<name>-hires-adetailer.png`, always png — that naming is what the pending scan keys on,
so re-running only ever does new work), and the list rescans itself after each
batch. A finished set doesn't appear; to load a folder anyway — to redo a set,
or one that keeps images outside a `Tests` folder — paste its path into the
**📂 Load Folder** box (re-runs land as `<name>-hires-1.png` /
`<name>-hires-adetailer-1.png`, originals are never overwritten).

## In forge link slots

In a [forge link](https://github.com/amiiari/forge-link) slot (`FORGELINK_SLOT` set), both
tabs are a front end: each dropped image is sent to the owner's Forge, which runs it with
the same code as here and sends it back without saving anything. The slot saves it in its
own outputs (`<name>-hires.png` in txt2img, `<name>-adetailer.png` in img2img).

- Each image goes through forge link's gate (`shared.forgelink.run_on_host`): its prompt
  (with every character edit and ADetailer prompt) and its final size are checked
  against the blocklist and size cap first, it waits its turn in the GPU queue (one turn
  per image, so others can generate in between), and the infotext that comes back is
  checked again.
- Batch ADetailer offers the owner's ADetailer units, read from her Forge.
- Nothing that touches the owner's disk is there: no test folders, folder loading,
  save-beside-the-source, `-base` copy, prompt export or first-revision lookup. Those are
  hidden, and refused server-side (folder listing returns nothing, scan roots are empty)
  whatever a client sends. The filename filter is off too.
- The owner's endpoints are `POST /batch-processing/v1/hires`, `POST
  /batch-processing/v1/adetailer` and `GET /batch-processing/v1/adetailer-units` on her
  Forge (local only, registered only where `FORGELINK_SLOT` isn't set).

## Settings

Under **Settings → Batch ADetailer** and **Settings → Batch Hires-Fix** (each
tab has its own section):

- **Output Directory** — custom save location (empty = the default img2img /
  txt2img output dir)
- **Test-folder scan roots** — semicolon-separated directories searched (3
  levels deep) for `Tests` folders by that tab's Test Folders panel, e.g.
  `C:\art\Commissions;C:\art\Requests`. The same roots are used to find
  drag-dropped images back on disk, so *save into each image's own folder* only
  works for images living under one of them
- **Skip Failed Images and Continue** — keep going when one image fails
  (default on)
- **Repair Unresolvable LoRA Names in Prompts** (ADetailer section) — the LoRA
  repair described above; one switch covers both tabs (default on)

## License

MIT
