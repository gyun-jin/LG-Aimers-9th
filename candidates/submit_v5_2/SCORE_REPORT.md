# SCORE_REPORT

## Version

- Version: `v5-2`
- Server score: `957.6800554874`

## Submission Record

`5-2/submit.zip` was submitted and returned `957.6800554874`.

## Local Validation

The final build report recorded:

- OOF mean Brier: `0.24696139`
- OOF worst Brier: `0.24986192`
- feature count: `118`
- model weights: CatBoost `0.8`, LightGBM `0.2`
- calibration: Platt

## Improvement Over Previous Versions

The v5 baseline already performed well, but v5-2 added prior-season Trackman summaries. Server score improved from the prior known v5 score `938.2526739958` to `957.6800554874`.

## Why Trackman Helped

Trackman summaries add pitcher-level physical and pitch-mix context that official row-wise game state does not fully encode:

- release speed stability
- spin-rate profile
- release position stability
- extension
- fastball/breaking/offspeed tendency
- mapping/history confidence

These features let the model distinguish pitchers with similar game context but different historical mechanics and pitch profiles.

## Limitations

- Trackman mapping quality varies by pitcher.
- Low-history pitchers rely heavily on priors.
- OOF gains were small; server gain was larger than local validation suggested.
- Deep models were not part of v5-2.

## Next Improvements

- Combine Trackman features with FT-Transformer/TabM embeddings.
- Add Trackman-aware stacking or gating by mapping confidence.
- Improve calibration beyond fixed Platt calibration.
- Search fold-specific or group-specific ensemble weights.
- Test Trackman feature subsets separately for CatBoost and LightGBM.
