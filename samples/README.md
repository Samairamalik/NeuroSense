# Sample files for testing and demonstration

Ready-to-use inputs for every modality, so you can run NeuroSense without
hunting for files. Run them all at once:

```bash
python samples/run_demo.py
```

Or upload them through the web app:

```bash
python app.py        # then open http://127.0.0.1:5000
```

---

## What is here

| File | Modality | True label | Source |
|---|---|---|---|
| `mri/normal_01 … normal_05` | MRI | normal | Project MRI dataset |
| `mri/parkinson_01 … parkinson_05` | MRI | parkinson | Project MRI dataset |
| `handwriting/spiral_01.png` | Handwriting | **unknown** | Bundled with the repo |
| `handwriting/wave_01.png` | Handwriting | **unknown** | Bundled with the repo |
| `voice/voice_01__sample.wav` | Voice | **unknown** | Bundled with the repo |

Each MRI filename keeps its original name after the `__`, so the acquisition
sequence stays visible — this matters, as explained below.

The ten MRI images were drawn at random with a fixed seed (42), five per class.
**Nothing was cherry-picked**: every image the random draw selected is included,
whether the model gets it right or not.

---

## Before you present these numbers

### The MRI result is a scanner artefact, not a medical finding

`run_demo.py` reports **10/10 correct** on MRI, with normal scans at ~1–5% and
Parkinson's scans at ~98–100%. That looks excellent. It is not.

The two classes in the dataset were acquired on **two different scanners**, and
nothing in the images overlaps:

**Image dimensions are completely disjoint**

| Class | Dimensions present in the dataset |
|---|---|
| normal | 320×250, 256×232, 192×192, 448×364, 320×260 |
| parkinson | 128×128, 256×256, 560×560 |

Not one dimension appears in both classes.

**Sequence names are disjoint too**

| Class | Sequences |
|---|---|
| normal | `t1_fl3d_sag_p4_iso`, `ep2d_diff_3scan_trace_p2`, `t2_tse_tra`, `SWI_Images`, `Mag_Images`, `Pha_Images`, `mIP_Images(SW)` |
| parkinson | `sDW_SSh_SENSE`, `DUAL_TSE`, `T1W_FFE`, `T2W_FLAIR`, `T2W_TSE`, `Survey_SHC` |

Those are two different vendors' conventions — `ep2d` and `p2/p4_iso` are
Siemens, `SENSE` and `DUAL_TSE` are Philips.

The model flattens each scan to raw pixel intensities, so field of view,
interpolation and contrast differences between the two machines are trivially
separable. **The model has almost certainly learned which scanner produced the
image, not whether the patient has Parkinson's.** Give it a healthy scan from
the Philips machine and it would very likely say "parkinson".

This is a textbook confound, and it is the first thing an examiner will ask
about once they see 100% confidence. Say it before they do — presenting the
flaw and how you found it is a far stronger position than presenting the 10/10
and being asked to explain it.

To fix it properly you would need both classes imaged on the same scanner with
the same protocol, or to control for acquisition explicitly.

### Most of these images were used to train the model

The MRI model was fitted on this same dataset folder with an 80/20 split, so
roughly 8 of these 10 images were almost certainly in the training set. The
model has seen them before. Accuracy measured on training data is always
optimistic, independent of the scanner problem above.

### Handwriting and voice have no ground truth

The spiral, wave and voice files are the only ones that exist in this project,
and none carries a label. Running them shows the pipeline works end to end —
it shows nothing about accuracy. Note also that all three score "elevated",
so there is no healthy example to demonstrate contrast with.

### The voice model is not fully reproducible

The script that trained `random_forest_tuned.pkl` is not in the repository. The
feature names and ordering were recovered from the model itself, but the
training-time definitions of jitter, shimmer and HNR could not be verified. See
[`../docs/VOICE_BUG.md`](../docs/VOICE_BUG.md).

---

## What this demo does legitimately show

- All four models load and run end to end.
- The full web interface works: upload, per-modality scoring, fusion, error
  handling.
- The handwriting pipeline produces sensible, differing scores on two different
  drawings (88.3% spiral, 74.5% wave).
- The voice pipeline extracts the correct 15 features in the correct order and
  reports interpretable clinical measures alongside the score.
- The MRI model separates the two dataset classes perfectly — which, read
  correctly, is a finding about the dataset rather than about the disease.

Those are real engineering results. The MRI accuracy figure is not a clinical
one, and should not be presented as though it were.

---

## Suggested order for a live demo

1. **Start with handwriting** — the clearest story. Spiral and wave score
   differently, and the drawn spiral motif in the interface explains what HOG
   is measuring.
2. **Then voice** — the waveform renders, and the clinical measures (jitter,
   shimmer, HNR) connect the model to real dysphonia markers.
3. **Then combined** — shows per-modality scores first and the fused score
   second, with the uncalibrated-fusion caveat visible on screen.
4. **Finish with MRI and the confound.** Show the 10/10, then the dimension
   table above, and explain why the number is not what it appears to be.
   Finding this is the most interesting result in the project.
