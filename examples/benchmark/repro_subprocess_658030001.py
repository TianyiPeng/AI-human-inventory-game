"""Reproduce the exact subprocess call used by benchmark_all_strategies for instance 658030001."""
import os
import sys
import subprocess
from pathlib import Path

# Same as benchmark_all_strategies
BASE_DIR = Path(__file__).parent.parent.parent
script_path = str(BASE_DIR / "examples" / "or_csv_demo.py")
instance_dir = str(BASE_DIR / "examples" / "benchmark" / "real_trajectory" / "lead_time_0" / "658030001")
test_file = os.path.join(instance_dir, "test.csv")
train_file = os.path.join(instance_dir, "train.csv")
base_dir = str(BASE_DIR)
cmd = [
    sys.executable,
    script_path,
    "--demand-file", test_file,
    "--promised-lead-time", "0",
    "--real-instance-train", train_file,
]
print("CMD:", cmd)
print("CWD:", base_dir)
print("Running...")
result = subprocess.run(
    cmd,
    cwd=base_dir,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=7200,
    stdin=subprocess.DEVNULL,
)
print("Return code:", result.returncode)
print("stdout length:", len(result.stdout or ""))
print("stderr length:", len(result.stderr or ""))
print("stdout repr (first 500):", repr((result.stdout or "")[:500]))
print("stderr repr (first 500):", repr((result.stderr or "")[:500]))
