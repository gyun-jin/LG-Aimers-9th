#!/usr/bin/env bash
set -euo pipefail

zip_path="${1:-/work/submission.zip}"
data_path="${2:-/work/data-source}"
timeout_seconds="${3:-600}"

python - <<'PY'
import importlib.metadata as md
import os
import platform

names = ["torch", "pandas", "numpy", "scipy", "scikit-learn"]
print("[container environment]")
print("OS:", platform.platform())
print("Python:", platform.python_version())
for name in names:
    print(f"{name}:", md.version(name))

import torch
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("GPU VRAM GiB:", round(props.total_memory / 1024**3, 2))
print("CPU affinity:", len(os.sched_getaffinity(0)))
PY

exec python /opt/checker/checker.py "$zip_path" \
    --data-dir "$data_path" \
    --timeout "$timeout_seconds"
