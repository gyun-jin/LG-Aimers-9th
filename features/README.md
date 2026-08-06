# Feature Ideas

Store one simple file for each reusable tabular feature idea, for example
`age_features.py` or `date_features.py`.

When a model uses a feature, copy its finalized logic into that model folder's
`train.py` and `script.py`. This keeps every model candidate independently
runnable and makes the final ZIP straightforward.
