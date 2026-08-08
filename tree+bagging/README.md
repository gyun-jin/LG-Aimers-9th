# Tree + Bagging baseline

`train.py` fits a submission-ready `DecisionTreeClassifier` ensemble using
`BaggingClassifier`. It validates on 2019-2023 -> 2024, applies sigmoid
probability calibration, then refits the tree ensemble on all labelled data.

```powershell
pip install -r tree+bagging\requirements.txt
python tree+bagging\train.py
python tree+bagging\script.py --input data\test.csv --output tree+bagging\prediction.csv
```

The model uses 18 selected raw features and 5 leakage-safe derived features.
Training creates `model/model.pkl`, `validation_metrics.json`, and a readable
`tree_preview.png` visualization of the first bagged tree.
