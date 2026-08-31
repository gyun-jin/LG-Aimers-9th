# TabM + CatBoost OOF blend report

- Rows: 746,504
- Residual correlation: 0.997233
- Selected weights: CatBoost 0.923871, TabM 0.076129
- Selected progressive-Platt mean Brier: 0.24689575
- Selected progressive-Platt worst Brier: 0.24987796
- Rolling nested mean Brier: 0.24690369

## Baselines

| model | raw mean | raw worst | progressive mean | progressive worst |
|---|---:|---:|---:|---:|
| catboost | 0.24693607 | 0.24991173 | 0.24691014 | 0.24990388 |
| tabm | 0.24813619 | 0.25244986 | 0.24834311 | 0.25283734 |

## Rolling nested selection

| validation season | CatBoost | TabM | Brier |
|---:|---:|---:|---:|
| 2022 | 1.00 | 0.00 | 0.24315017 |
| 2023 | 1.00 | 0.00 | 0.24990388 |
| 2024 | 0.95 | 0.05 | 0.24765703 |

## Weight grid

| CatBoost | TabM | raw mean | raw worst | progressive mean | progressive worst |
|---:|---:|---:|---:|---:|---:|
| 0.000000 | 1.000000 | 0.24813619 | 0.25244986 | 0.24834311 | 0.25283734 |
| 0.050000 | 0.950000 | 0.24800900 | 0.25217903 | 0.24821448 | 0.25257376 |
| 0.100000 | 0.900000 | 0.24788887 | 0.25192334 | 0.24808262 | 0.25229760 |
| 0.150000 | 0.850000 | 0.24777582 | 0.25168281 | 0.24795490 | 0.25203068 |
| 0.200000 | 0.800000 | 0.24766984 | 0.25145743 | 0.24783193 | 0.25177442 |
| 0.250000 | 0.750000 | 0.24757094 | 0.25124719 | 0.24771429 | 0.25153020 |
| 0.300000 | 0.700000 | 0.24747910 | 0.25105211 | 0.24760257 | 0.25129930 |
| 0.350000 | 0.650000 | 0.24739434 | 0.25087217 | 0.24749735 | 0.25108294 |
| 0.400000 | 0.600000 | 0.24731665 | 0.25070739 | 0.24739916 | 0.25088219 |
| 0.450000 | 0.550000 | 0.24724604 | 0.25055776 | 0.24730853 | 0.25069806 |
| 0.500000 | 0.500000 | 0.24718250 | 0.25042328 | 0.24722595 | 0.25053136 |
| 0.550000 | 0.450000 | 0.24712603 | 0.25030395 | 0.24715185 | 0.25038282 |
| 0.600000 | 0.400000 | 0.24707663 | 0.25019977 | 0.24708663 | 0.25025298 |
| 0.650000 | 0.350000 | 0.24703430 | 0.25011073 | 0.24703063 | 0.25014224 |
| 0.700000 | 0.300000 | 0.24699905 | 0.25003685 | 0.24698411 | 0.25005085 |
| 0.750000 | 0.250000 | 0.24697087 | 0.24997812 | 0.24694728 | 0.24997890 |
| 0.800000 | 0.200000 | 0.24694977 | 0.24993454 | 0.24692027 | 0.24992633 |
| 0.850000 | 0.150000 | 0.24693573 | 0.24990611 | 0.24690312 | 0.24989294 |
| 0.900000 | 0.100000 | 0.24692877 | 0.24989283 | 0.24689581 | 0.24987839 |
| 0.923871 | 0.076129 | 0.24692794 | 0.24989184 | 0.24689575 | 0.24987796 |
| 0.950000 | 0.050000 | 0.24692888 | 0.24989471 | 0.24689821 | 0.24988223 |
| 1.000000 | 0.000000 | 0.24693607 | 0.24991173 | 0.24691014 | 0.24990388 |

## Cautions

- CatBoost OOF uses seed 42 only as a proxy; the submitted v5_8 artifact averages five seeds.
- TabM OOF uses one epoch because CPU training takes 7.5-13.8 minutes per fold and epoch.
- CatBoost OOF uses feature state stored by the final full-training v5_8 artifact.
- CatBoost OOF uses official train.csv; original v5_8 validation removed 116 hand-mismatch rows.
- Both base learners use the target fold for early stopping, matching their existing validation style.
- Global Platt metrics are in-sample diagnostics and are not used for weight selection.
