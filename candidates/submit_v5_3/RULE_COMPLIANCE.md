# RULE COMPLIANCE

- [x] Official competition data only
- [x] No external data
- [x] No remote API
- [x] No forbidden pretrained model or weights
- [x] No use of other rows in `test.csv`
- [x] No current-pitch future information
- [x] No current-pitch actual location or course
- [x] No current-pitch actual call, result, or `control_success`
- [x] No current-pitch actual pitch type
- [x] No current-pitch Trackman measurements
- [x] No 2025 Trackman data
- [x] Trackman is used as historical pitcher summary only
- [x] Current-season features use row-owned `asof_*` values plus train-time constants only
- [x] `submit.zip` top-level structure is `model/`, `script.py`, `requirements.txt`
- [x] Original train/test/Trackman CSVs are not committed
