"""LG Aimers 9th code-submission preflight checker.

Usage:
    python checker.py path/to/submission.zip
    python checker.py path/to/submission.zip --data-dir path/to/open/data

The checker extracts the ZIP into a fresh temporary directory, recreates the
official ``./data``, ``./model`` and ``./output`` layout, runs ``script.py``
without arguments, and validates ``./output/submission.csv``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


MAX_ZIP_BYTES = 10 * 1024**3
MAX_EXTRACTED_BYTES = 32 * 1024**3
REQUIRED_ROOT_FILES = {"script.py", "requirements.txt"}
REQUIRED_COLUMNS = ["row_id", "control_success"]
JUNK_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
JUNK_PARTS = {"__MACOSX", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}

# Versions copied from the supplied evaluation-environment notes. A mismatch
# is a warning because requirements.txt may intentionally replace them.
SERVER_PACKAGES = {
    "torch": "2.7.1+cu128",
    "pandas": "2.0.3",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "scikit-learn": "1.8.0",
    "joblib": "1.5.3",
}

IMPORT_TO_PACKAGE = {
    "catboost": "catboost",
    "joblib": "joblib",
    "lightgbm": "lightgbm",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "xgboost": "xgboost",
}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)

    def passed(self, message: str) -> None:
        self.passes.append(message)
        print(f"[PASS] {message}")

    def warned(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def failed(self, message: str) -> None:
        self.errors.append(message)
        print(f"[ERROR] {message}")


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def normalize_member(name: str) -> str:
    return name.replace("\\", "/")


def validate_zip(zip_path: Path, report: Report) -> list[zipfile.ZipInfo]:
    print("\n[1/7] ZIP structure and size")
    if zip_path.suffix.lower() != ".zip":
        report.failed(f"Submission file must have a .zip extension: {zip_path.name!r}")
    else:
        report.passed(f"ZIP filename accepted: {zip_path.name}")

    zip_size = zip_path.stat().st_size
    if zip_size > MAX_ZIP_BYTES:
        report.failed(f"ZIP exceeds 10 GiB: {human_bytes(zip_size)}")
    else:
        report.passed(f"ZIP size: {human_bytes(zip_size)}")

    try:
        archive = zipfile.ZipFile(zip_path)
        bad_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        report.failed(f"Invalid ZIP: {exc}")
        return []

    with archive:
        infos = archive.infolist()
        if bad_member:
            report.failed(f"Corrupted ZIP member: {bad_member}")
        else:
            report.passed("ZIP CRC check")

        names = [normalize_member(info.filename) for info in infos]
        normalized = set(names)
        extracted_size = sum(info.file_size for info in infos)
        if extracted_size > MAX_EXTRACTED_BYTES:
            report.failed(f"Extracted size exceeds 32 GiB: {human_bytes(extracted_size)}")
        else:
            report.passed(f"Expected extracted size: {human_bytes(extracted_size)}")

        for required in REQUIRED_ROOT_FILES:
            if required not in normalized:
                report.failed(f"Missing ZIP-root file: {required}")
            else:
                report.passed(f"ZIP-root file exists: {required}")

        model_files = [
            name
            for info, name in zip(infos, names)
            if name.startswith("model/") and name != "model/" and not info.is_dir()
        ]

        if "model/" in normalized:
            report.passed("Explicit model/ directory entry exists")
        elif model_files:
            report.passed("model/ directory is implicitly represented by its file entries")

        if not model_files:
            report.failed("No model artifact file exists under ZIP-root model/")
        else:
            report.passed(f"Model artifacts: {model_files}")

        allowed_roots = {"script.py", "requirements.txt", "model"}
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        unexpected_roots = sorted(roots - allowed_roots)
        if unexpected_roots:
            report.failed(f"Unexpected ZIP-root entries: {unexpected_roots}")
        else:
            report.passed("No extra top-level wrapper directory")

        for info, name in zip(infos, names):
            path = PurePosixPath(name)
            if name.startswith("/") or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
                report.failed(f"Unsafe ZIP path: {name!r}")
            if "\\" in info.filename:
                report.failed(f"Windows backslash used in ZIP member: {info.filename!r}")
            if path.name in JUNK_NAMES or any(part in JUNK_PARTS for part in path.parts):
                report.failed(f"Junk/cache file included: {name}")
            if name in {"script.py", "requirements.txt", "model/"} or name in model_files:
                if info.external_attr == 0:
                    # ZIPs created by Windows tools such as Compress-Archive may
                    # omit Unix mode bits entirely. This does not prevent the
                    # evaluator from extracting the member, and script.py is
                    # launched through Python rather than as an executable.
                    # Extraction and required-file checks below are the checks
                    # that determine whether the archive is usable.
                    report.warned(
                        f"ZIP member has no Unix permission metadata: {name}; "
                        "safe extraction will be verified."
                    )

    return infos


def parse_requirements(path: Path, report: Report) -> set[str]:
    print("\n[3/7] requirements.txt")
    packages: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith(("-e ", "git+", "http://", "https://")) or " @ http" in lowered:
            report.failed(f"Unsafe/network requirement at line {line_number}: {line}")
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*(?:==\s*([^;\s]+))?", line)
        if not match:
            report.warned(f"Could not parse requirement line {line_number}: {line}")
            continue
        package = match.group(1).lower().replace("_", "-")
        version = match.group(2)
        packages.add(package)
        if version is None:
            report.warned(f"Dependency is not pinned: {package}")
        if package in SERVER_PACKAGES and version and version != SERVER_PACKAGES[package]:
            report.warned(
                f"{package}: requested {version}, documented server base {SERVER_PACKAGES[package]}"
            )
    if packages:
        report.passed(f"Parsed {len(packages)} dependencies: {sorted(packages)}")
    else:
        report.passed("No additional dependencies requested")
    return packages


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def analyze_script(path: Path, packages: set[str], report: Report) -> None:
    print("\n[4/7] script.py static analysis")
    source = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(source, filename="script.py")
    except SyntaxError as exc:
        report.failed(f"script.py syntax error: {exc}")
        return

    report.passed("script.py parses successfully")
    lowered = source.lower().replace("\\", "/")
    path_patterns = {
        "absolute Windows path": r"[a-z]:/users/|[a-z]:/content/",
        "Colab path": r"/content/drive/|/content/",
        "user-home path": r"/home/[^/]+/|~/",
        "parent-directory data access": r"\.\./(?:data|open|train|test)",
    }
    for label, pattern in path_patterns.items():
        if re.search(pattern, lowered):
            report.failed(f"Detected {label} in script.py")

    expected_tokens = {
        "data directory": "data",
        "test input": "test.csv",
        "model directory": "model",
        "output directory": "output",
        "submission output": "submission.csv",
    }
    for label, token in expected_tokens.items():
        if token not in lowered:
            report.failed(f"Official path token missing from script.py: {label} ({token})")
        else:
            report.passed(f"Official path referenced: {token}")
    if "sample_submission.csv" not in lowered:
        report.warned(
            "script.py does not read ./data/sample_submission.csv; runtime output must still match it exactly."
        )
    else:
        report.passed("Official sample path referenced: data/sample_submission.csv")

    imported: set[str] = set()
    suspicious_calls: list[tuple[int, str]] = []
    network_prefixes = (
        "requests.",
        "urllib.request.",
        "wget.",
        "gdown.",
        "huggingface_hub.",
        "torch.hub.",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name.startswith(network_prefixes):
                suspicious_calls.append((node.lineno, name))
            if name in {"subprocess.run", "subprocess.Popen", "os.system"}:
                report.warned(f"External process call at script.py:{node.lineno}: {name}")

    if suspicious_calls:
        for line, name in suspicious_calls:
            report.failed(f"Network/download call at script.py:{line}: {name}")
    else:
        report.passed("No known network/download calls")

    for module, package in IMPORT_TO_PACKAGE.items():
        if module in imported and package not in packages and package not in SERVER_PACKAGES:
            report.failed(f"Imported package missing from requirements.txt: {package}")

    # control_success itself is not suspicious here because the official output
    # column must use that name. Flag only unambiguous post-pitch naming patterns.
    suspicious_features = [
        token for token in ("post_pitch", "pitch_result") if token in lowered
    ]
    if suspicious_features:
        report.warned(
            "Review possible target/post-pitch references: " + ", ".join(suspicious_features)
        )


def safe_extract(zip_path: Path, destination: Path, report: Report) -> bool:
    print("\n[2/7] isolated extraction and official layout")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                normalized = normalize_member(info.filename)
                path = PurePosixPath(normalized)
                if normalized.startswith("/") or ".." in path.parts:
                    report.failed(f"Extraction blocked for unsafe path: {normalized}")
                    return False
            archive.extractall(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        report.failed(f"ZIP extraction failed: {exc}")
        return False

    required = [destination / "script.py", destination / "requirements.txt", destination / "model"]
    for path in required:
        if not path.exists():
            report.failed(f"Missing after extraction: {path.relative_to(destination)}")
        else:
            report.passed(f"Extracted path exists: ./{path.relative_to(destination)}")
    model_files = [path for path in (destination / "model").rglob("*") if path.is_file()]
    if not model_files:
        report.failed("No model file is visible at ./model after extraction")
    else:
        report.passed(
            "Extracted model files: "
            + ", ".join(str(path.relative_to(destination)) for path in model_files)
        )
    return not report.errors


def discover_data_dir(explicit: str | None, zip_path: Path) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            Path.cwd() / "data",
            zip_path.parent / "data",
            zip_path.parent.parent / "data",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "test.csv").is_file() and (resolved / "sample_submission.csv").is_file():
            return resolved
    return None


def stage_official_data(source: Path, root: Path, report: Report) -> bool:
    data_dir = root / "data"
    output_dir = root / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    for name in ("test.csv", "sample_submission.csv"):
        source_file = source / name
        if not source_file.is_file():
            report.failed(f"Official simulation input is missing: {source_file}")
            return False
        shutil.copy2(source_file, data_dir / name)
        report.passed(f"Server path staged: ./data/{name}")
    report.passed("Server output path staged: ./output/")
    return True


def print_captured_output(value: str) -> None:
    """Print child-process output without letting a legacy console abort checks."""
    encoding = sys.stdout.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe_value)


def run_submission(
    root: Path,
    python_executable: str,
    timeout: int,
    report: Report,
) -> bool:
    print("\n[5/7] server simulation: python script.py")
    started = time.perf_counter()
    # The official examples may print Unicode status markers (for example, ✅).
    # A captured pipe otherwise inherits the Windows console code page (often
    # CP949), turning a successful inference into an unrelated encoding error.
    # The evaluator runs Python non-interactively, so use UTF-8 for the child
    # process and decode its captured streams with the same encoding.
    child_env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        completed = subprocess.run(
            [python_executable, "script.py"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        report.failed(f"script.py exceeded timeout: {timeout}s")
        return False
    except OSError as exc:
        report.failed(f"Could not start script.py: {exc}")
        return False

    elapsed = time.perf_counter() - started
    if completed.stdout.strip():
        print("--- stdout ---")
        print_captured_output(completed.stdout.rstrip())
    if completed.stderr.strip():
        print("--- stderr ---")
        print_captured_output(completed.stderr.rstrip())
    if completed.returncode != 0:
        report.failed(f"script.py exit code {completed.returncode} after {elapsed:.2f}s")
        return False
    report.passed(f"script.py completed in {elapsed:.2f}s / {timeout}s")
    return True


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_output(root: Path, report: Report) -> None:
    print("\n[6/7] output/submission.csv")
    output_path = root / "output" / "submission.csv"
    if not output_path.is_file():
        report.failed("Missing required output: ./output/submission.csv")
        return
    report.passed("Output exists: ./output/submission.csv")

    output_columns, output_rows = read_csv_rows(output_path)
    sample_columns, sample_rows = read_csv_rows(root / "data" / "sample_submission.csv")
    test_columns, test_rows = read_csv_rows(root / "data" / "test.csv")

    if output_columns != REQUIRED_COLUMNS:
        report.failed(f"Output columns must be {REQUIRED_COLUMNS}, got {output_columns}")
    else:
        report.passed(f"Output columns: {output_columns}")
    if sample_columns[:2] != REQUIRED_COLUMNS:
        report.failed(f"Sample submission columns are unexpected: {sample_columns}")
    if len(output_rows) != len(sample_rows) or len(output_rows) != len(test_rows):
        report.failed(
            f"Row mismatch: output={len(output_rows)}, sample={len(sample_rows)}, test={len(test_rows)}"
        )
    else:
        report.passed(f"Output row count: {len(output_rows):,}")

    output_ids = [row.get("row_id", "") for row in output_rows]
    sample_ids = [row.get("row_id", "") for row in sample_rows]
    test_ids = [row.get("row_id", "") for row in test_rows]
    if output_ids != sample_ids:
        report.failed("Output row_id values/order do not exactly match sample_submission.csv")
    else:
        report.passed("Output row_id values/order match sample_submission.csv")
    if sample_ids != test_ids:
        report.failed("sample_submission.csv row_id order does not match test.csv")
    if len(output_ids) != len(set(output_ids)):
        report.failed("Duplicate row_id values in output")
    else:
        report.passed("No duplicate row_id values")

    probabilities: list[float] = []
    for index, row in enumerate(output_rows, 2):
        raw = row.get("control_success", "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            report.failed(f"Non-numeric probability at CSV line {index}: {raw!r}")
            return
        if not math.isfinite(value):
            report.failed(f"NaN/inf probability at CSV line {index}: {raw!r}")
            return
        if not 0.0 <= value <= 1.0:
            report.failed(f"Probability outside [0,1] at CSV line {index}: {value}")
            return
        probabilities.append(value)
    if probabilities:
        report.passed(
            "Probabilities finite and within [0,1]: "
            f"min={min(probabilities):.6f}, max={max(probabilities):.6f}, "
            f"mean={sum(probabilities) / len(probabilities):.6f}"
        )


def print_final(report: Report, sandbox: Path | None) -> int:
    print("\n[7/7] final report")
    print("=" * 64)
    print(f"Passes   : {len(report.passes)}")
    print(f"Warnings : {len(report.warnings)}")
    print(f"Errors   : {len(report.errors)}")
    if sandbox:
        print(f"Sandbox  : {sandbox}")
    if report.errors:
        print("\nNOT READY TO SUBMIT")
        for item in report.errors:
            print(f"  - {item}")
        return 1
    print("\nREADY TO SUBMIT")
    if report.warnings:
        print("Review warnings before upload.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LG Aimers submission ZIP checker")
    parser.add_argument("zip_path", type=Path, help="Path to the submission ZIP")
    parser.add_argument(
        "--data-dir",
        help="Directory containing official test.csv and sample_submission.csv",
    )
    parser.add_argument("--python", default=sys.executable, help="Python used to run script.py")
    parser.add_argument("--timeout", type=int, default=600, help="Inference timeout in seconds")
    parser.add_argument("--keep-temp", action="store_true", help="Keep extracted sandbox")
    parser.add_argument("--skip-run", action="store_true", help="Only run static/ZIP checks")
    args = parser.parse_args()

    zip_path = args.zip_path.expanduser().resolve()
    report = Report()
    print("LG AIMERS Submission Checker")
    print("=" * 64)
    print(f"ZIP: {zip_path}")
    print("Official layout: ./data, ./model, ./output, ./script.py")
    if not zip_path.is_file():
        report.failed(f"ZIP does not exist: {zip_path}")
        return print_final(report, None)

    infos = validate_zip(zip_path, report)
    if not infos:
        return print_final(report, None)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        sandbox = Path(tempfile.mkdtemp(prefix="lg_aimers_submission_"))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="lg_aimers_submission_")
        sandbox = Path(temporary.name)

    safe_extract(zip_path, sandbox, report)
    if (sandbox / "requirements.txt").is_file():
        packages = parse_requirements(sandbox / "requirements.txt", report)
    else:
        packages = set()
    if (sandbox / "script.py").is_file():
        analyze_script(sandbox / "script.py", packages, report)

    if args.skip_run:
        report.warned("Runtime and output checks were skipped by --skip-run")
        result = print_final(report, sandbox if args.keep_temp else None)
        if temporary:
            temporary.cleanup()
        return result

    data_source = discover_data_dir(args.data_dir, zip_path)
    if data_source is None:
        report.failed(
            "Could not find a data directory containing both test.csv and "
            "sample_submission.csv. Pass --data-dir explicitly."
        )
    else:
        print(f"\nSimulation data source: {data_source}")
        if stage_official_data(data_source, sandbox, report) and not report.errors:
            if run_submission(sandbox, args.python, args.timeout, report):
                validate_output(sandbox, report)
        elif report.errors:
            report.warned("Runtime skipped because preflight errors must be fixed first")

    result = print_final(report, sandbox if args.keep_temp else None)
    if temporary:
        temporary.cleanup()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
