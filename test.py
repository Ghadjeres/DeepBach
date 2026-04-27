import os
import sys
import time
import inspect
import traceback

# =============================================================================
#  CONFIG — edit all parameters here
# =============================================================================

# --- Files / paths -----------------------------------------------------------
# Voice-0 melody file.  Leave "" to open a file dialog.
MELODY_FILE     = ""

# Pretrained model weight directory.  Leave "" for the package default (<package>/models/).
MODELS_DIR      = ""

# Custom MIDI corpus directory for the custom_midi_dir interface test.
# Leave "" to skip that branch.
CUSTOM_MIDI_DIR = ""

# All output files are written here.
OUTPUT_DIR      = "./test_outputs"

# --- harmonize() parameters --------------------------------------------------
HARMONIZE_ITERATIONS      = 5000   # Gibbs sampling iterations (higher = slower, better)
HARMONIZE_TEMPERATURE     = 0.5    # Sampling temperature: 1.0 standard, <1 conservative
HARMONIZE_BATCH_PER_VOICE = 64     # Parallel Gibbs proposals per step
HARMONIZE_MELODY_VOICE    = 0      # Fixed voice index: 0 = soprano

# --- generate_from_scratch() parameters --------------------------------------
SCRATCH_SEQUENCE_TICKS    = 64     # Generation length in 16th-note ticks (64 = 4 bars)
SCRATCH_ITERATIONS        = 2000   # Gibbs sampling iterations
SCRATCH_TEMPERATURE       = 1.0    # Sampling temperature

# --- Flow switches -----------------------------------------------------------
# Skip generate_from_scratch to save time.
SKIP_SCRATCH = True

# Set True to actually run training (very slow).
# False = signature check only.
RUN_TRAIN    = False

# train_from_scratch parameters (used only when RUN_TRAIN=True)
TRAIN_EPOCHS     = 20    # Full convergence ceiling
TRAIN_BATCH_SIZE = 512   # RTX 4050 / 8 GB VRAM

# finetune parameters (used only when RUN_TRAIN=True and weights exist)
FINETUNE_EPOCHS        = 5     # Standard fine-tune recipe
FINETUNE_BATCH_SIZE    = 256
FINETUNE_LR            = 1e-4  # Lower LR to avoid destroying pretrained features
FINETUNE_VOICE_INDICES = [0, 1, 2, 3]  # Fine-tune all four voices

# =============================================================================
#  Test infrastructure — no need to edit below this line
# =============================================================================

OUTPUT_DIR      = os.path.abspath(OUTPUT_DIR)
MODELS_DIR      = os.path.abspath(MODELS_DIR)      if MODELS_DIR      else None
CUSTOM_MIDI_DIR = os.path.abspath(CUSTOM_MIDI_DIR) if CUSTOM_MIDI_DIR else None
os.makedirs(OUTPUT_DIR, exist_ok=True)

PASS = "✓ PASS"
FAIL = "✗ FAIL"
SKIP = "⚠ SKIP"
results: list[tuple[str, str, str]] = []


def section(title: str):
    w = 72
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def record(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    icon = status.split()[0]
    print(f"  [{icon}] {name}" + (f"  —  {detail}" if detail else ""))


def run_test(name: str, fn, *a, **kw):
    """Run fn(*a, **kw), record elapsed time and pass/fail. Never raises."""
    try:
        t0 = time.perf_counter()
        result = fn(*a, **kw)
        elapsed = time.perf_counter() - t0
        record(name, PASS, f"{elapsed:.1f}s")
        return result, True
    except Exception as exc:
        record(name, FAIL, str(exc))
        traceback.print_exc()
        return None, False


# =============================================================================
# Step 0 — Select Voice-0 melody file
# =============================================================================
section("Step 0 — Select Voice-0 melody file")

melody_path = None

if MELODY_FILE:
    melody_path = os.path.abspath(MELODY_FILE)
    print(f"  Using path from config: {melody_path}")
else:
    print("  MELODY_FILE is empty — opening file dialog...")
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Select Voice-0 melody file (MIDI / MusicXML)",
            filetypes=[
                ("Music files",   "*.mid *.midi *.xml *.mxl *.musicxml"),
                ("MIDI files",    "*.mid *.midi"),
                ("MusicXML",      "*.xml *.mxl *.musicxml"),
                ("All files",     "*.*"),
            ],
        )
        root.destroy()
        melody_path = chosen if chosen else None
    except Exception as e:
        print(f"  File dialog error: {e}")

MELODY_OK = bool(melody_path and os.path.exists(melody_path))
if MELODY_OK:
    record("Select melody file", PASS, melody_path)
else:
    record("Select melody file", SKIP, "No file provided — harmonize tests will be skipped")
    print("  ⚠ Set MELODY_FILE in the config section to enable harmonize().")

# =============================================================================
# Step 1 — Import package
# =============================================================================
section("Step 1 — Import deepbach_pytorch package")

try:
    import deepbach_pytorch as db
    record("import deepbach_pytorch", PASS)
except ImportError as e:
    record("import deepbach_pytorch", FAIL, str(e))
    print("\n[Fatal] Install the package first: pip install deepbach_pytorch")
    sys.exit(1)

EXPECTED_SYMBOLS = [
    "harmonize", "generate_from_scratch", "train_from_scratch", "finetune",
    "build_dataset", "build_model", "create_model", "load_model",
    "check_pretrained_weights", "get_device_info",
    "DeepBach", "DatasetManager", "ChoraleDataset",
    "FermataMetadata", "TickMetadata", "KeyMetadata", "PACKAGE_ROOT",
]
missing = [s for s in EXPECTED_SYMBOLS if not hasattr(db, s)]
if missing:
    record("__all__ completeness", FAIL, f"Missing: {missing}")
else:
    record("__all__ completeness", PASS,
           f"All {len(EXPECTED_SYMBOLS)} symbols exposed")

# =============================================================================
# Step 2 — Utility API
# =============================================================================
section("Step 2 — Utility API: get_device_info / check_pretrained_weights")

device_info, _ = run_test("get_device_info()", db.get_device_info)
if device_info:
    print(f"         device type  : {device_info['device_type']}")
    print(f"         device name  : {device_info['device_name']}")
    print(f"         CUDA available: {device_info['cuda_available']}")

weight_status, _ = run_test("check_pretrained_weights()",
                            db.check_pretrained_weights,
                            models_dir=MODELS_DIR)
WEIGHTS_OK = False
if weight_status:
    WEIGHTS_OK = weight_status["complete"]
    print(f"         models_dir     : {weight_status['models_dir']}")
    print(f"         found/required : {weight_status['found_weights']}"
          f" / {weight_status['required_weights']}")
    if not WEIGHTS_OK:
        print(f"         missing voices : {weight_status.get('missing_voices', [])}")
        print("  ⚠ Weights incomplete — inference tests will be skipped.")

# =============================================================================
# Step 3 — build_dataset()  (with custom_midi_dir branch)
# =============================================================================
section("Step 3 — build_dataset()  (custom_midi_dir interface)")

dataset, ds_ok = run_test("build_dataset() — Bach Chorales", db.build_dataset)
if ds_ok and dataset:
    print(f"         type        : {type(dataset).__name__}")
    print(f"         num voices  : {getattr(dataset, 'num_voices', '?')}")

if CUSTOM_MIDI_DIR:
    print(f"\n  custom_midi_dir = {CUSTOM_MIDI_DIR}")
    custom_dataset, _ = run_test(
        "build_dataset(custom_midi_dir=...)",
        db.build_dataset,
        custom_midi_dir=CUSTOM_MIDI_DIR,    # custom_midi_dir interface
    )
    if custom_dataset:
        print(f"         num voices  : {getattr(custom_dataset, 'num_voices', '?')}")
else:
    record("build_dataset(custom_midi_dir=...)", SKIP, "CUSTOM_MIDI_DIR not configured")

# =============================================================================
# Step 4 — create_model / build_model(load_weights=False)
# =============================================================================
section("Step 4 — create_model() / build_model(load_weights=False)")

bare_model, bm_ok = run_test(
    "create_model()",
    db.create_model,
    dataset=dataset if ds_ok else None,
    models_dir=MODELS_DIR,
)
if bm_ok and bare_model:
    print(f"         type        : {type(bare_model).__name__}")
    print(f"         voice models: {len(getattr(bare_model, 'voice_models', []))}")

run_test(
    "build_model(load_weights=False)",
    db.build_model,
    dataset=dataset if ds_ok else None,
    models_dir=MODELS_DIR,
    load_weights=False,
)

# =============================================================================
# Step 5 — load_model / build_model(load_weights=True)
# =============================================================================
section("Step 5 — load_model() / build_model(load_weights=True)")

if not WEIGHTS_OK:
    record("load_model()",                  SKIP, "weights incomplete")
    record("build_model(load_weights=True)", SKIP, "weights incomplete")
else:
    run_test("load_model()",
             db.load_model,
             dataset=dataset if ds_ok else None,
             models_dir=MODELS_DIR)

    run_test("build_model(load_weights=True)",
             db.build_model,
             dataset=dataset if ds_ok else None,
             models_dir=MODELS_DIR,
             load_weights=True)

# =============================================================================
# Step 6 — harmonize(): fix Voice-0, generate alto / tenor / bass
# =============================================================================
section("Step 6 — harmonize()  Voice-0 fixed → generate three harmonising voices")


def _print_score_info(score):
    """Print basic part statistics for a music21 Score."""
    if score is None:
        return
    try:
        for i, part in enumerate(score.parts):
            notes = part.flatten().notes
            print(f"         voice {i}: {part.partName or '(unnamed)'}"
                  f"  /  {len(notes)} notes")
    except Exception:
        pass


if not WEIGHTS_OK:
    record("harmonize() — voice-0 fixed, generate voices 1-3", SKIP, "weights incomplete")
    record("harmonize() — custom_midi_dir",                     SKIP, "weights incomplete")
elif not MELODY_OK:
    record("harmonize() — voice-0 fixed, generate voices 1-3", SKIP, "no melody file")
    record("harmonize() — custom_midi_dir",                     SKIP, "no melody file")
else:
    # --- 6-A: standard usage: soprano fixed, generate alto / tenor / bass ----
    out_a = os.path.join(OUTPUT_DIR, "harmonized_voice0.xml")
    print(f"\n  [params] melody_voice={HARMONIZE_MELODY_VOICE}"
          f"  keep_melody=True"
          f"  temperature={HARMONIZE_TEMPERATURE}"
          f"  iterations={HARMONIZE_ITERATIONS}"
          f"  batch_per_voice={HARMONIZE_BATCH_PER_VOICE}")

    score_a, ok_a = run_test(
        "harmonize() — voice-0 fixed, generate voices 1-3",
        db.harmonize,
        input_file           = melody_path,
        output_path          = out_a,
        num_iterations       = HARMONIZE_ITERATIONS,
        temperature          = HARMONIZE_TEMPERATURE,   # temperature interface
        batch_size_per_voice = HARMONIZE_BATCH_PER_VOICE,
        melody_voice         = HARMONIZE_MELODY_VOICE,
        keep_melody          = True,
        models_dir           = MODELS_DIR,
    )
    if ok_a:
        print(f"  ✓ Saved: {out_a}")
        _print_score_info(score_a)

    # --- 6-B: custom_midi_dir branch -----------------------------------------
    if CUSTOM_MIDI_DIR:
        out_b = os.path.join(OUTPUT_DIR, "harmonized_custom_corpus.xml")
        run_test(
            "harmonize() — custom_midi_dir",
            db.harmonize,
            input_file      = melody_path,
            output_path     = out_b,
            num_iterations  = HARMONIZE_ITERATIONS,
            temperature     = HARMONIZE_TEMPERATURE,
            melody_voice    = HARMONIZE_MELODY_VOICE,
            keep_melody     = True,
            models_dir      = MODELS_DIR,
            custom_midi_dir = CUSTOM_MIDI_DIR,          # custom_midi_dir interface
        )
    else:
        record("harmonize() — custom_midi_dir", SKIP, "CUSTOM_MIDI_DIR not configured")

# =============================================================================
# Step 7 — generate_from_scratch(): all four voices, no melody input
# =============================================================================
section("Step 7 — generate_from_scratch()  fully random four-voice chorale")

if not WEIGHTS_OK:
    record("generate_from_scratch() — all voices", SKIP, "weights incomplete")
    record("generate_from_scratch() — custom_midi_dir", SKIP, "weights incomplete")
elif SKIP_SCRATCH:
    record("generate_from_scratch() — all voices", SKIP, "SKIP_SCRATCH=True")
    record("generate_from_scratch() — custom_midi_dir", SKIP, "SKIP_SCRATCH=True")
else:
    # --- 7-A: standard usage -------------------------------------------------
    out_s = os.path.join(OUTPUT_DIR, "scratch_four_voices.xml")
    print(f"\n  [params] sequence_length_ticks={SCRATCH_SEQUENCE_TICKS}"
          f"  temperature={SCRATCH_TEMPERATURE}"
          f"  iterations={SCRATCH_ITERATIONS}")

    score_s, ok_s = run_test(
        "generate_from_scratch() — all voices",
        db.generate_from_scratch,
        sequence_length_ticks = SCRATCH_SEQUENCE_TICKS,  # sequence_length_ticks interface
        num_iterations        = SCRATCH_ITERATIONS,
        temperature           = SCRATCH_TEMPERATURE,     # temperature interface
        models_dir            = MODELS_DIR,
    )
    if ok_s and score_s:
        try:
            score_s.write("musicxml", fp=out_s)
            print(f"  ✓ Saved: {out_s}")
            _print_score_info(score_s)
        except Exception as e:
            print(f"  Save failed: {e}")

    # --- 7-B: custom_midi_dir branch -----------------------------------------
    if CUSTOM_MIDI_DIR:
        run_test(
            "generate_from_scratch() — custom_midi_dir",
            db.generate_from_scratch,
            sequence_length_ticks = SCRATCH_SEQUENCE_TICKS,
            num_iterations        = SCRATCH_ITERATIONS,
            temperature           = SCRATCH_TEMPERATURE,
            models_dir            = MODELS_DIR,
            custom_midi_dir       = CUSTOM_MIDI_DIR,     # custom_midi_dir interface
        )
    else:
        record("generate_from_scratch() — custom_midi_dir", SKIP,
               "CUSTOM_MIDI_DIR not configured")

# =============================================================================
# Step 8 — train_from_scratch / finetune
# =============================================================================
section("Step 8 — train_from_scratch() / finetune()  (training API)")

if not RUN_TRAIN:
    print("  RUN_TRAIN=False: signature check only, no actual training.\n")
    for fn_name in ["train_from_scratch", "finetune"]:
        fn = getattr(db, fn_name, None)
        if callable(fn):
            record(f"{fn_name}() callable", PASS)
            sig = inspect.signature(fn)
            print(f"  {fn_name} signature:")
            for pname, p in sig.parameters.items():
                default = ("(required)" if p.default is inspect.Parameter.empty
                           else f"default={p.default!r}")
                print(f"    {pname:24s} {default}")
            print()
        else:
            record(f"{fn_name}() callable", FAIL, "not found or not callable")
else:
    # --- 8-A: train from scratch (full run) ----------------------------------
    train_dir = os.path.join(OUTPUT_DIR, "trained_models")
    run_test(
        f"train_from_scratch() — {TRAIN_EPOCHS} epochs",
        db.train_from_scratch,
        num_epochs      = TRAIN_EPOCHS,
        batch_size      = TRAIN_BATCH_SIZE,
        models_dir      = train_dir,
        custom_midi_dir = CUSTOM_MIDI_DIR,          # custom_midi_dir interface
    )

    # --- 8-B: finetune (requires existing weights) ---------------------------
    if WEIGHTS_OK:
        run_test(
            f"finetune() — {FINETUNE_EPOCHS} epochs, all voices",
            db.finetune,
            num_epochs      = FINETUNE_EPOCHS,
            batch_size      = FINETUNE_BATCH_SIZE,
            lr              = FINETUNE_LR,
            voice_indices   = FINETUNE_VOICE_INDICES,
            models_dir      = MODELS_DIR,
            custom_midi_dir = CUSTOM_MIDI_DIR,      # custom_midi_dir interface
        )
    else:
        record("finetune()", SKIP, "weights incomplete")

# =============================================================================
# Step 9 — Low-level class and constant export check
# =============================================================================
section("Step 9 — Low-level class and constant export check")

for symbol in ["DeepBach", "DatasetManager", "ChoraleDataset",
               "FermataMetadata", "TickMetadata", "KeyMetadata"]:
    cls = getattr(db, symbol, None)
    record(f"db.{symbol}", PASS if cls else FAIL,
           str(cls) if cls else "not found")

pkg_root = getattr(db, "PACKAGE_ROOT", None)
record("db.PACKAGE_ROOT",
       PASS if (pkg_root and os.path.isdir(pkg_root)) else FAIL,
       str(pkg_root))

try:
    db.FermataMetadata()
    db.TickMetadata(subdivision=4)
    db.KeyMetadata()
    record("Metadata instantiation (Fermata / Tick / Key)", PASS)
except Exception as e:
    record("Metadata instantiation", FAIL, str(e))

# =============================================================================
# Summary
# =============================================================================
section("Test Summary")

total   = len(results)
passed  = sum(1 for _, s, _ in results if s == PASS)
failed  = sum(1 for _, s, _ in results if s == FAIL)
skipped = sum(1 for _, s, _ in results if s == SKIP)

print(f"\n  Total {total}   ✓ passed {passed}   ✗ failed {failed}   ⚠ skipped {skipped}\n")

if failed:
    print("  ── Failed tests " + "─" * 48)
    for name, status, detail in results:
        if status == FAIL:
            print(f"  {FAIL}  {name}")
            if detail:
                print(f"          {detail}")
    print()

if passed == total - skipped:
    print("  🎉 All executable tests passed.")
else:
    print("  ⚠  Some tests failed — check the log above.")

print(f"\n  Output directory: {OUTPUT_DIR}")
print()
