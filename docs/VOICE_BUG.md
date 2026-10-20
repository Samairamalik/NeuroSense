# The voice pipeline was feeding the model the wrong features

## What was wrong

`models/random_forest_tuned.pkl` was fitted on a pandas DataFrame, so it carries
the names of its training columns in `feature_names_in_`:

```
mfcc_16_std, mfcc_18_std, mfcc_19_std, mfcc_0_std,  mfcc_1_mean,
mfcc_15_std, mfcc_1_std,  energy_std,  mfcc_11_std, mfcc_10_std,
mfcc_3_std,  mfcc_2_std,  mfcc_8_std,  mfcc_6_std,  mfcc_13_std
```

That is a feature-selection result — the top 15 by importance, dominated by MFCC
standard deviations.

Both `app.py` and `sound/test_sound.py` instead passed a hand-written list:

```
f0_mean, f0_std, f0_median, jitter_local, jitter_rap, shimmer_local, hnr,
mfcc_0_mean, mfcc_1_mean, mfcc_0_std, mfcc_1_std,
energy_mean, energy_std, zcr_mean, zcr_std
```

Since the input was a plain NumPy array, sklearn could not check the names and
matched purely by position. The result, measured on `sound/sample.wav`:

**0 of 15 features landed in the correct position.**

| Slot | Was fed | Value | Model expected | Correct value |
|---|---|---|---|---|
| 0 | `f0_mean` | 131.60 | `mfcc_16_std` | 2.48 |
| 3 | `jitter_local` | 0.0002 | `mfcc_0_std` | 14.41 |
| 6 | `hnr` | 0.00 | `mfcc_1_std` | 10.01 |
| 7 | `mfcc_0_mean` | −145.27 | `energy_std` | 0.018 |

A tree trained to split `energy_std` at ~0.02 was receiving −145. Every voice
prediction the app ever produced was decided by values with no relationship to
the features the splits were learned on.

`sound/test_sound.py` even carried the comment
`# ADJUST THIS LIST based on your actual training features!` — the list was
never adjusted.

Two secondary bugs compounded it, both in `app.py` only:

- `jitter_local` was not normalised by the mean period, making it ~100× smaller
  than the training-time definition; `jitter_rap` was a copy of it rather than
  the real RAP formula.
- `hnr` was hardcoded to `0` with the comment "simplified for stability",
  discarding one of the most clinically meaningful dysphonia measures.
- `shimmer_local` used an unnormalised RMS difference instead of the normalised
  amplitude-envelope definition.

Effect on the sample file: the old pipeline reported **77.91%**, the corrected
one reports **86.85%**. The old number was not a worse estimate — it was not an
estimate at all.

## Why it happened

The same feature-extraction logic existed in three places (the Flask app, the
standalone script, and the training code) with no shared source. Two copies
drifted and nothing detected it, because a positional array silently accepts any
15 numbers.

## The fix

1. One canonical implementation in `src/voice_features.py`. The app and the
   script both import it; neither computes features itself.
2. Jitter, shimmer and HNR implemented properly (normalised ratios, real HNR in
   dB).
3. **Feature selection is driven by the model**, not by a hand-written list:

   ```python
   names = model.feature_names_in_
   frame = pd.DataFrame([[features[n] for n in names]], columns=names)
   ```

   Passing a named DataFrame also lets sklearn validate the columns, so a
   mismatch now raises instead of silently mispredicting. Retraining on a
   different feature set requires no code change.
4. `build_feature_frame` raises a clear error listing any feature the model needs
   that extraction did not produce.

## Still outstanding

The voice model's **training script is not in the repository**, so the exact
training-time definitions of jitter, shimmer and HNR cannot be verified from
code — only the feature *names* and ordering are recoverable. The implementations
here follow the standard definitions, but if the original training script still
exists it should be committed and the definitions checked against it. Until then,
treat absolute voice probabilities with caution; the ordering fix is certain, the
scaling of the four non-MFCC features is not.
