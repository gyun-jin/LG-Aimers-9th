# CatBoost 95% + TabM 5% submission probe

This package is the conservative server probe selected by rolling temporal OOF.
It blends the raw five-seed `submit_v5_8` CatBoost prediction with a one-epoch
full-data TabM prediction, then applies a Platt calibrator fitted to the aligned
95:5 raw OOF blend.

The package is experimental: the rolling mean Brier improvement over CatBoost
alone was about 0.0000064, so it should not replace `submit_v5_8` without a
better server score.
