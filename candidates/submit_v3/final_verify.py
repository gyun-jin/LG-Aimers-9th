"""최종 submit.zip의 구조, 독립 실행, 성능, 누수 정적 검사를 재현한다."""

# [추가 구현]
# 목적: CODEX_TASK 최종 검증 15개 항목을 기계적으로 감사해 JSON 저장
from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from calibration import apply_calibrator
from inference_utils import predict_bundle

EXPECTED_ZIP_FILES = {
    "model/ensemble_config.json",
    "model/calibration_model.joblib",
    "model/final_model.joblib",
    "model/feature_schema.json",
    "script.py",
    "requirements.txt",
}


def validate_submission(path: Path, test: pd.DataFrame, sample: pd.DataFrame) -> dict:
    output = pd.read_csv(path)
    assert list(output.columns) == ["row_id", "control_success"]
    assert len(output) == len(test) == len(sample)
    assert output["row_id"].is_unique
    assert output["row_id"].tolist() == sample["row_id"].tolist()
    assert set(output["row_id"]) == set(test["row_id"])
    values = output["control_success"].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0)).all()
    return {
        "rows": len(output),
        "columns": list(output.columns),
        "id_set_match": True,
        "id_order_match": True,
        "prediction_min": float(values.min()),
        "prediction_max": float(values.max()),
        "prediction_mean": float(values.mean()),
        "finite_and_in_range": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submit.zip"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--solution-dir", type=Path, default=Path("solution"))
    parser.add_argument("--clean-python", type=Path)
    parser.add_argument("--clean-install-seconds", type=float, default=None)
    args = parser.parse_args()

    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", encoding="utf-8-sig")
    zip_path = args.zip.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        assert names == EXPECTED_ZIP_FILES, (names, EXPECTED_ZIP_FILES)
        uncompressed_bytes = sum(item.file_size for item in archive.infolist())

    if args.clean_python:
        # venv의 python symlink 자체를 보존해야 시스템 Python으로 잘못 해석되지 않는다.
        python = args.clean_python if args.clean_python.is_absolute() else Path.cwd() / args.clean_python
    else:
        python = Path(__import__("sys").executable)
    before_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    with tempfile.TemporaryDirectory(prefix="codex-submit-verify-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        # 실제 제출 규칙대로 open/만 제공해 경로 의존성을 검증한다.
        data = root / "open"
        data.mkdir()
        shutil.copy2(args.data_dir / "test.csv", data / "test.csv")
        shutil.copy2(args.data_dir / "sample_submission.csv", data / "sample_submission.csv")
        started = time.perf_counter()
        result = subprocess.run(
            [str(python), "script.py"], cwd=root, capture_output=True, text=True, check=False
        )
        subprocess_wall = time.perf_counter() - started
        if result.returncode != 0:
            raise RuntimeError(result.stdout + "\n" + result.stderr)
        submission = validate_submission(root / "output" / "submission.csv", test, sample)
        timing_match = re.search(r"load=([0-9.]+)s inference=([0-9.]+)s", result.stdout)
        if timing_match is None:
            raise ValueError(f"추론 시간 로그를 해석할 수 없음: {result.stdout}")
        sample_model_load = float(timing_match.group(1))
        sample_inference = float(timing_match.group(2))

    script_text = (args.solution_dir / "script.py").read_text(encoding="utf-8")
    internet_patterns = [r"https?://", r"\brequests\b", r"\burllib\b", r"\bsocket\b"]
    test_aggregation_patterns = [
        r"\.groupby\(",
        r"\.rolling\(",
        r"\.expanding\(",
        r"\.value_counts\(",
        r"target.?encod",
    ]
    internet_hits = [pattern for pattern in internet_patterns if re.search(pattern, script_text, re.I)]
    aggregation_hits = [
        pattern for pattern in test_aggregation_patterns if re.search(pattern, script_text, re.I)
    ]
    assert not internet_hits
    assert not aggregation_hits

    schema = json.loads((args.solution_dir / "model" / "feature_schema.json").read_text())
    forbidden_after_pitch = {
        "control_success",
        "actual_pitch_type",
        "pitch_result",
        "plate_x",
        "plate_z",
        "catcher_target",
        "target_direction",
        "direction_match",
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
        "extension",
        "rel_height",
        "rel_side",
        "zone_speed",
    }
    forbidden_present = sorted(forbidden_after_pitch & set(schema["input_columns"]))
    assert not forbidden_present

    # 배포 행 수와 같은 실제 프레임으로 추론 시간을 측정한다. 통계 fit은 하지 않는다.
    n = 245_789
    large = test.iloc[np.arange(n) % len(test)].reset_index(drop=True)
    large["row_id"] = [f"BENCH_{index:06d}" for index in range(n)]
    started = time.perf_counter()
    bundle = joblib.load(args.solution_dir / "model" / "final_model.joblib")
    calibrator = joblib.load(args.solution_dir / "model" / "calibration_model.joblib")
    benchmark_load = time.perf_counter() - started
    started = time.perf_counter()
    benchmark_predictions = apply_calibrator(calibrator, predict_bundle(bundle, large))
    benchmark_inference = time.perf_counter() - started
    assert len(benchmark_predictions) == n and np.isfinite(benchmark_predictions).all()

    model_bytes = sum(
        path.stat().st_size for path in (args.solution_dir / "model").rglob("*") if path.is_file()
    )
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report = {
        "python": str(python),
        "clean_requirements_install_seconds_cached": args.clean_install_seconds,
        "sample_submission": submission,
        "sample_model_load_seconds": sample_model_load,
        "sample_inference_seconds": sample_inference,
        "sample_process_wall_seconds": subprocess_wall,
        "sample_child_max_rss_bytes": int(max(before_children, child_rss)),
        "benchmark_rows": n,
        "benchmark_model_load_seconds": benchmark_load,
        "benchmark_inference_seconds": benchmark_inference,
        "benchmark_rows_per_second": n / benchmark_inference,
        "benchmark_max_rss_bytes": int(max_rss),
        "model_bytes": model_bytes,
        "zip_bytes": zip_path.stat().st_size,
        "zip_uncompressed_bytes": uncompressed_bytes,
        "zip_files": sorted(EXPECTED_ZIP_FILES),
        "external_internet_code_hits": internet_hits,
        "test_internal_aggregation_code_hits": aggregation_hits,
        "current_pitch_post_outcome_input_columns": forbidden_present,
        "cpu_inference_supported": True,
        "all_checks_passed": True,
    }
    output = args.solution_dir / "output" / "final_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
