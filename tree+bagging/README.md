# Tree + Bagging baseline

## Evaluation-environment compatibility

`requirements.txt` is pinned to the documented evaluation-server versions:

```text
numpy==1.26.4
pandas==2.0.3
scikit-learn==1.8.0
scipy==1.15.3
joblib==1.5.3
```

The existing `model/model.pkl` and `submit.zip` were created with the older
NumPy 2.3.5 / scikit-learn 1.7.2 environment. They are not compatible with the
versions above. After changing environments, run `train.py` to create a new
`model/model.pkl`, run inference once, and rebuild the submission ZIP. Do not
package the existing model with the updated requirements.

`train.py` fits a submission-ready `DecisionTreeClassifier` ensemble using
`BaggingClassifier`. It validates on 2019-2023 -> 2024, then refits the tree
ensemble on all labelled data. Calibration is intentionally omitted because
the submission checklist requires calibration to be learned from past OOF predictions.

```powershell
pip install -r tree+bagging\requirements.txt
python tree+bagging\train.py
python tree+bagging\script.py --input data\test.csv --output tree+bagging\prediction.csv
```

The model uses 18 selected raw features and 5 leakage-safe derived features.
Training creates `model/model.pkl`, `validation_metrics.json`, and a readable
`tree_preview.png` visualization of the first bagged tree.
