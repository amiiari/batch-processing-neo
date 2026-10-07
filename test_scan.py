"""Self-check for both stages' scan logic and the shared core. Run: python test_scan.py
(Forge modules are stubbed out — this only exercises the pure filesystem code.)
"""
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

for _m in ("gradio", "PIL", "modules", "modules.infotext_utils", "modules_forge"):
    sys.modules[_m] = MagicMock()

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                            # batch_adetailer_shared
sys.path.insert(0, os.path.join(_here, "scripts"))
import batch_adetailer_shared as bshared
import batch_adetailer as bad
import batch_hires_fix as bhf


def touch(*parts, data=b""):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ── Hires stage (runs first): bases pending until a -hires sibling exists ──
with tempfile.TemporaryDirectory() as root:
    tests = os.path.join(root, "Commission 1 - A", "Tests")
    touch(tests, "1r1.png")                                     # pending base
    touch(tests, "2r1.png"); touch(tests, "2r1-hires.png")  # hires-fixed -> done
    touch(tests, "3r1.png"); touch(tests, "3r1-hires.png")
    touch(tests, "3r1-hires-base.png")                      # later-stage outputs -> not bases
    touch(tests, "3r1-hires-adetailer.png")
    touch(tests, "4r1.png"); touch(tests, "4r1-adetailer.png")      # old adetailer-first chain -> left alone
    touch(tests, "10r1.png")                                    # pending; after 1r1, natural order
    touch(tests, "2r1-hires-1.png")                         # collision copy -> ignored
    touch(tests, "notes.txt")                                   # not an image -> ignored
    pending = [os.path.basename(p) for p in bhf._pending_bases(tests)]
    assert pending == ["1r1.png", "10r1.png"], pending

    # "Load Folder" takes every base, done or not — but still never a variant.
    bases = [os.path.basename(p) for p in bhf._base_images(tests)]
    assert bases == ["1r1.png", "2r1.png", "3r1.png", "4r1.png", "10r1.png"], bases

    touch(root, "Commission 2 - B", "Tests", "1r1.png")         # 1 pending
    done = os.path.join(root, "Commission 3 - C", "Tests")
    touch(done, "1r1.png"); touch(done, "1r1-hires.png")    # done -> hidden
    touch(root, "Commission 4 - D", "readme.txt")               # no Tests dir -> hidden

    # A set nested one level deeper (root/Commissions/<set>), as the real tree
    # has: sets sit both directly under the root and under a group folder.
    nested = os.path.join(root, "Commissions", "Commission 5 - E")
    touch(nested, "Tests", "1r1.png")
    touch(nested, "9r1.png")

    # One entry per set: its own folder + its Tests folder, counted together.
    # Folders without a Tests marker (scratch, references) aren't sets; work
    # below Tests (Finished archives) isn't scanned.
    set_a = os.path.join(root, "Commission 1 - A")
    touch(set_a, "8r1.png")                # + the 2 pending in its Tests -> 3
    touch(tests, "Finished", "7r1.png")
    touch(root, "random", "junk.png")
    set_b = os.path.join(root, "Commission 2 - B")   # pending only in Tests

    # Request sets have no Tests subfolder — being inside Requests is the marker.
    req = os.path.join(root, "Requests", "Request 1 - X")
    touch(req, "1r1.png"); touch(req, "2r1.png")
    done_req = os.path.join(root, "Requests", "Request 2 - Y")
    touch(done_req, "1r1.png"); touch(done_req, "1r1-hires.png")  # done -> hidden

    # Only the highest rN revision of each image number counts, double digits
    # included (1r2 < 1r13); names outside the NrM convention pass through.
    touch(req, "1r2.png"); touch(req, "1r13.png")
    touch(req, "flower2.png")
    latest = [os.path.basename(p) for p in bhf._base_images(req)]
    assert latest == ["1r13.png", "2r1.png", "flower2.png"], latest

    bshared.shared.opts.batch_hires_fix_scan_roots = root + ";" + os.path.join(root, "missing")
    choices = bshared.scan_test_folders(bhf.STAGE)
    assert [c[1] for c in choices] == [set_a, set_b, nested, req], choices
    assert "(3 to do)" in choices[0][0], choices
    assert "(1 to do)" in choices[1][0] and "(2 to do)" in choices[2][0], choices
    assert "(3 to do)" in choices[3][0], choices  # 1r13 + 2r1 + flower2

    # Drag-dropped copies resolve back to their originals: same name in two sets,
    # told apart by content; a name that exists nowhere stays a temp path.
    a = touch(tests, "9r1.png", data=b"AAA")
    b = touch(nested, "9r1.png", data=b"BBB")
    cache = os.path.join(root, "gradio-cache")
    bshared.is_dragged_temp_copy = lambda p: p.startswith(cache)
    drops = [touch(cache, "h1", "9r1.png", data=b"BBB"),      # -> b
             touch(cache, "h2", "9r1.png", data=b"AAA"),      # -> a
             touch(cache, "h3", "nowhere.png", data=b"CCC")]  # -> unresolved
    resolved, notes = bshared.resolve_dropped_paths(drops, bhf.STAGE)
    assert resolved == [b, a, drops[2]], resolved
    assert len(notes) == 1 and "1 dropped file(s) not found" in notes[0], notes
    assert "Batch Hires-Fix" in notes[0], notes  # the note names this tab's settings

    # Export → import round-trip: configs follow the filename, not the path.
    cfg = [0, "prompt A", "neg A", 0.3, 0.4, 1.0] * 2
    msg = bad._export_prompts({a: cfg}, [a], root)
    assert "1 image(s)" in msg, msg
    exported = [f for f in os.listdir(root) if f.endswith(".json")]
    assert len(exported) == 1, exported
    new_home = touch(root, "elsewhere", "9r1.png")
    store, *_, msg = bad._import_prompts(
        os.path.join(root, exported[0]), {new_home: ["x"] * len(cfg)},
        [new_home], None, 2)
    assert store[new_home] == cfg, store
    assert "1 of 1" in msg, msg

    # Same filename in two sets: each keeps its own entry, and an ambiguous
    # filename-only match is refused rather than guessed.
    data = {bshared.export_key(a): "A", bshared.export_key(b): "B"}
    assert len(data) == 2, data
    assert bshared.lookup_export(data, a) == "A" and bshared.lookup_export(data, b) == "B"
    assert bshared.lookup_export(data, new_home) is None
    assert bshared.lookup_export({"9r1.png": "old"}, new_home) == "old"  # pre-set-key export


# ── ADetailer stage (runs second): -hires inputs pending until a -adetailer/-edited sibling ──
with tempfile.TemporaryDirectory() as root:
    tests = os.path.join(root, "Commission 1 - A", "Tests")
    touch(tests, "1r1.png"); touch(tests, "1r1-hires.png")     # pending
    touch(tests, "2r1.png"); touch(tests, "2r1-hires.png")
    touch(tests, "2r1-hires-adetailer.png")                        # detailed -> done
    touch(tests, "2r1-base.png")                               # lanczos twin -> not an input
    touch(tests, "3r1.png"); touch(tests, "3r1-hires.png")
    touch(tests, "3r1-hires-edited.png")                       # edited past -> done
    touch(tests, "4r1.png")                                        # base only -> not ready yet
    touch(tests, "10r1.png"); touch(tests, "10r1-hires.jpg")   # pending; after 1r1, natural order
    touch(tests, "1r1-hires-1.png")                            # collision copy -> ignored
    touch(tests, "notes.txt")                                      # not an image -> ignored
    pending = [os.path.basename(p) for p in bad._pending_hires(tests)]
    assert pending == ["1r1-hires.png", "10r1-hires.jpg"], pending

    # "Load Folder" takes every -hires input, done or not — but never an
    # output (-hires-adetailer/-base/-edited) or a collision copy.
    inputs = [os.path.basename(p) for p in bad._hires_images(tests)]
    assert inputs == ["1r1-hires.png", "2r1-hires.png", "3r1-hires.png",
                      "10r1-hires.jpg"], inputs

    touch(root, "Commission 2 - B", "Tests", "1r1-hires.png")  # 1 pending
    done = os.path.join(root, "Commission 3 - C", "Tests")
    touch(done, "1r1-hires.png"); touch(done, "1r1-hires-adetailer.png")  # done -> hidden
    touch(root, "Commission 4 - D", "readme.txt")                  # no Tests dir -> hidden

    nested = os.path.join(root, "Commissions", "Commission 5 - E")
    touch(nested, "Tests", "1r1-hires.png")
    touch(nested, "9r1-hires.png")

    set_a = os.path.join(root, "Commission 1 - A")
    touch(set_a, "8r1-hires.png")      # + the 2 pending in its Tests -> 3
    touch(tests, "Finished", "7r1-hires.png")
    touch(root, "random", "junk-hires.png")
    set_b = os.path.join(root, "Commission 2 - B")   # pending only in Tests

    req = os.path.join(root, "Requests", "Request 1 - X")
    touch(req, "2r1.png"); touch(req, "2r1-hires.png")
    done_req = os.path.join(root, "Requests", "Request 2 - Y")
    touch(done_req, "1r1-hires.png")
    touch(done_req, "1r1-hires-adetailer.png")                     # done -> hidden

    touch(req, "1r1-hires.png"); touch(req, "1r2-hires.png")
    touch(req, "1r13-hires.png")
    touch(req, "flower2-hires.png")
    latest = [os.path.basename(p) for p in bad._hires_images(req)]
    assert latest == ["1r13-hires.png", "2r1-hires.png",
                      "flower2-hires.png"], latest

    bshared.shared.opts.batch_adetailer_scan_roots = root + ";" + os.path.join(root, "missing")
    choices = bshared.scan_test_folders(bad.STAGE)
    assert [c[1] for c in choices] == [set_a, set_b, nested, req], choices
    assert "(3 to do)" in choices[0][0], choices
    assert "(1 to do)" in choices[1][0] and "(2 to do)" in choices[2][0], choices
    assert "(3 to do)" in choices[3][0], choices  # 1r13 + 2r1 + flower2

    a = touch(tests, "9r1-hires.png", data=b"AAA")
    b = touch(nested, "9r1-hires.png", data=b"BBB")
    cache = os.path.join(root, "gradio-cache")
    bshared.is_dragged_temp_copy = lambda p: p.startswith(cache)
    drops = [touch(cache, "h1", "9r1-hires.png", data=b"BBB"),  # -> b
             touch(cache, "h2", "9r1-hires.png", data=b"AAA"),  # -> a
             touch(cache, "h3", "nowhere-hires.png", data=b"C")]  # -> unresolved
    resolved, notes = bshared.resolve_dropped_paths(drops, bad.STAGE)
    assert resolved == [b, a, drops[2]], resolved
    assert len(notes) == 1 and "1 dropped file(s) not found" in notes[0], notes
    assert "Batch ADetailer" in notes[0], notes


# ── LoRA tokens from other tools carry a subfolder and/or extension; Forge
# indexes by bare stem, so those must resolve to it. ──
class Net:
    def __init__(self, name): self.name = name
nets = MagicMock()
nets.available_networks = {"amiiari-anima-v5.3": Net("amiiari-anima-v5.3"),
                           "Dark_Slider_Anima": Net("Dark_Slider_Anima")}
nets.available_network_aliases = {}
assert bshared.resolve_lora_name("amiiari-anima-v5.3", nets) is None  # already fine
assert bshared.resolve_lora_name("amiiari-anima-v5.3.safetensors", nets) == "amiiari-anima-v5.3"
assert bshared.resolve_lora_name(r"Misc\Dark_Slider_Anima.safetensors", nets) == "Dark_Slider_Anima"
assert bshared.resolve_lora_name("gone.safetensors", nets) == ""


# ── Infotext inheritance: styles restored, Clip skip recovered from the raw
# text (parse always pops it), subseed strength / seed resize applied. ──
bshared.parse_generation_parameters = lambda text, skip: {
    "Prompt": "a prompt", "Negative prompt": "bad",
    "Styles array": ["My Style"],
    "Seed": "123", "Variation seed": "456",
    "Steps": "30", "CFG scale": "6.5",
    "Variation seed strength": "0.7",
    "Seed resize from-1": "640", "Seed resize from-2": "960",
    "Sampler": "Euler a", "Schedule type": "Karras",
}
p = SimpleNamespace(override_settings={})
bshared.apply_source_image_parameters(p, "a prompt\nSteps: 30, Clip skip: 2")
assert p.styles == ["My Style"], p.styles
assert p.steps == 30 and p.cfg_scale == 6.5
assert p.subseed_strength == 0.7, p.subseed_strength
assert (p.seed_resize_from_w, p.seed_resize_from_h) == (640, 960)
assert p.override_settings["CLIP_stop_at_last_layers"] == 2, p.override_settings
assert p.sampler_name == "Euler a" and p.scheduler == "Karras"


# ── Per-run cancel tokens: a cancel kills runs already started, never one
# started after (the old global flag was reset by any new run start). ──
AD, HR = bshared.ADETAILER_STAGE, bshared.HIRES_STAGE
t1 = bshared.start_run(AD)
assert not bshared.cancel_requested(AD, t1)
bshared.request_cancel(AD)
assert bshared.cancel_requested(AD, t1)
t2 = bshared.start_run(AD)
assert not bshared.cancel_requested(AD, t2)
assert bshared.cancel_requested(AD, t1)  # the old run stays cancelled

# ...and one tab's Cancel never stops the other tab's batch.
hr_run = bshared.start_run(HR)
bshared.request_cancel(AD)
assert not bshared.cancel_requested(HR, hr_run)
bshared.request_cancel(HR)
assert bshared.cancel_requested(HR, hr_run)

# ── Hires-Fix per-image prompts: "use the first revision's prompt" picks the
# LOWEST revision of the same image number, preferring the plain base file over
# its -adetailer/-hires variants. Only the latest revision is ever loaded into
# the tab, so this reads siblings straight off disk. ──
with tempfile.TemporaryDirectory() as root:
    for n in ("20r1.png", "20r1-adetailer.png", "20r2-adetailer.png",
              "20r3-adetailer.png", "20r10-adetailer.png", "5r1.png", "sketch.png"):
        touch(root, n)

    read = []
    bshared.image_prompt = lambda p: (read.append(os.path.basename(p)),
                                   "prompt of " + os.path.basename(p))[1]

    prompt, note = bhf._first_revision_prompt(os.path.join(root, "20r3-adetailer.png"))
    assert prompt == "prompt of 20r1.png", (prompt, note)
    assert read == ["20r1.png"], read  # the base, not its -adetailer twin

    # 20r10 must not be treated as a "20r1" prefix match, either way round.
    prompt, _ = bhf._first_revision_prompt(os.path.join(root, "20r10-adetailer.png"))
    assert prompt == "prompt of 20r1.png", prompt

    prompt, note = bhf._first_revision_prompt(os.path.join(root, "5r1.png"))
    assert prompt is None and "already" in note, (prompt, note)

    prompt, note = bhf._first_revision_prompt(os.path.join(root, "sketch.png"))
    assert prompt is None, (prompt, note)

    # An override is decided by membership, not truthiness: a cleared box means
    # "run with no prompt", not "fall back to the image's own".
    assert bhf._prompt_for({os.path.join(root, "20r1.png"): ""},
                           os.path.join(root, "20r1.png")) == ""
    assert bhf._prompt_for({}, os.path.join(root, "20r1.png")) == "prompt of 20r1.png"

# ── Prompt editing collapses to the finished image's last-step state for the
# ADetailer pass. Needs Forge's real grammar (lark + torch: run with Forge's venv). ──
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "prompt_parser", os.path.join(_here, "..", "..", "modules", "prompt_parser.py"))
    _pp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pp)
except ImportError as e:
    print(f"skipped prompt-editing checks ({e})")
else:
    sys.modules["modules"].prompt_parser = _pp
    fp = bshared.final_prompt
    assert fp("x [thing : thing3 : 7] y", 30) == "x  thing3  y", fp("x [thing : thing3 : 7] y", 30)
    assert fp("x [thing:7]", 30) == "x thing"
    assert fp("x [a::7]", 30) == "x "
    assert fp("x [a:b:0.3]", 30) == "x b"
    assert fp("x [a:b:40]", 30) == "x a"                     # never switched
    assert fp("[a:[b:c:5]:3]", 30) == "c"                     # nested
    assert fp("(a:1.2), [b|c], [d]", 30) == "(a:1.2), [b|c], [d]"  # untouched
    assert fp("b, [<lora:x:1> a::5]", 30) == "b, <lora:x:1>", fp("b, [<lora:x:1> a::5]", 30)  # LoRA was always on
    assert fp("<lora:y:0.5> no schedule", 30) == "<lora:y:0.5> no schedule"
    assert fp("a [unbalanced", 30) == "a [unbalanced"

    units = [{"ad_prompt": "", "ad_negative_prompt": "bad"},
             {"ad_prompt": "face, [PROMPT] [SEP] ", "ad_negative_prompt": ""},
             {"ad_prompt": "eyes", "ad_negative_prompt": ""}]
    bad._collapse_prompt_editing(units, SimpleNamespace(
        prompt="girl [smile:frown:7]", negative_prompt="ugly", steps=30))
    assert units[0] == {"ad_prompt": "girl frown", "ad_negative_prompt": "bad"}, units[0]
    assert units[1]["ad_prompt"] == "face, girl frown[SEP]girl frown", units[1]
    assert units[1]["ad_negative_prompt"] == "", units[1]  # no schedule -> ADetailer's own fallback
    assert units[2]["ad_prompt"] == "eyes", units[2]

    # Explicit face prompts keep their schedules and LoRAs in their own [SEP]
    # segments, exactly as regular ADetailer receives them.
    face_prompts = [
        "<lora:amiiari-anima-v5.3:0.95>, @amiiari, red hair, "
        "[very long hair : long hair : 0.5]",
        "<lora:amiiari-anima-v5.3:0.95>, @amiiari, [mature::14], "
        "black hair, pink eyes, [black eyeliner:7]",
        "<lora:amiiari-anima-v5.3:0.95>, @amiiari, [mature::7], "
        "dark brown hair, dark green eyes, [black eyeliner:7]",
    ]
    prompt = "\n\n[SEP]\n\n".join(face_prompts)
    negative = "[bad::7] [SEP] [worse:7] [SEP] worst"
    units = [{"ad_prompt": prompt, "ad_negative_prompt": negative}]
    bad._collapse_prompt_editing(units, SimpleNamespace(
        prompt="base [smile:frown:7]", negative_prompt="ugly", steps=30))
    assert units[0] == {"ad_prompt": prompt, "ad_negative_prompt": negative}, units[0]

    # Only inherited text is collapsed. Even in a mixed prompt, inherited
    # LoRAs stay in each face's segment and authored edits still run normally.
    units = [{"ad_prompt": "[PROMPT], [eyeliner:7] [SEP] [SKIP] [SEP] ",
              "ad_negative_prompt": "[PROMPT] [SEP] [bad::7]"}]
    bad._collapse_prompt_editing(units, SimpleNamespace(
        prompt="<lora:base:0.8>, girl [smile:frown:7]",
        negative_prompt="[ugly:worse:7]", steps=30))
    parts = bad._SEP_RE.split(units[0]["ad_prompt"])
    assert len(parts) == 3 and parts[1] == "[SKIP]", parts
    assert [part.count("<lora:base:0.8>") for part in parts] == [1, 0, 1], parts
    assert "[eyeliner:7]" in parts[0] and "girl frown" in parts[0], parts
    assert units[0]["ad_negative_prompt"] == "worse[SEP][bad::7]", units[0]

    # forge-stagehand: characters ride in the prompt as lines under the main prompt.
    assert bshared.has_characters("masterpiece, 2girls\n\nCharacter 1 (Ruby): girl\nCharacter 2 at 0.500 0 1 1: girl")
    assert bshared.has_characters("Character 3: girl")
    assert bshared.has_characters("base\n\nCharacter 1 (Ruby) at B2: girl")  # a grid cell
    # several places and an overlap share -- missing these gave C02 noise faces
    assert bshared.has_characters("Character 1 (kira) at 0.503 0.000 1.000 1.000 + 0.318 0.000 0.512 0.663: girl")
    assert bshared.has_characters("Character 2 (annie) at 0.000 0.233 0.324 1.000 + 0.327 0.650 1.000 1.000, share 70%: girl")
    assert bshared.has_characters("Character 2, share 30%: girl")
    assert not bshared.has_characters("a Character study\nCharacter design, 1girl")
    assert not bshared.has_characters("")

    # Scripts opt in to restoring their args from an image's infotext; the rest keep defaults.
    class Opts:  # restores itself
        args_from, args_to = 2, 4
        def title(self): return "opts"
        def args_from_infotext(self, params): return [params["a"], params["b"]]
    class Declines(Opts):
        args_from, args_to = 4, 5
        def args_from_infotext(self, params): return None
    class Wrong(Opts):
        args_from, args_to = 5, 6
        def args_from_infotext(self, params): return [1, 2]  # wrong length: ignored
    class Breaks(Opts):
        args_from, args_to = 6, 7
        def args_from_infotext(self, params): raise ValueError("no")
    class Plain:
        args_from, args_to = 1, 2
    args = [0, "p", "d1", "d2", "d3", "d4", "d5"]
    runner = SimpleNamespace(alwayson_scripts=[Plain(), Opts(), Declines(), Wrong(), Breaks()])
    bshared.replay_script_args(runner, args, {"a": "A", "b": "B"})
    assert args == [0, "p", "A", "B", "d3", "d4", "d5"], args

print("ok")
