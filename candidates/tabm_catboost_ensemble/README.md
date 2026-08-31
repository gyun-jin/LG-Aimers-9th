# TabM + CatBoost ensemble experiment

This folder builds aligned 2022, 2023, and 2024 temporal OOF predictions for
`submit_v5_8` CatBoost and `tabm_v1`, then selects a convex probability blend.

Run from the repository root with the `microtabpfn` conda environment:

```powershell
conda run -n microtabpfn python candidates/tabm_catboost_ensemble/run_oof.py
conda run -n microtabpfn python candidates/tabm_catboost_ensemble/analyze_blend.py
```

The OOF runner is resumable per model and fold. Its outputs are written under
`output/`.
