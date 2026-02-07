"""
Cleanup script: remove old benchmark results and logs so we can rerun with a new model.

What it deletes:
- For every instance directory (contains train.csv + test.csv) under:
  - examples/benchmark/synthetic_trajectory/**
  - examples/benchmark/real_trajectory/**
  It removes:
    - benchmark_results.json
    - llm_1.txt
    - llm_to_or_1.txt
    - or_1.txt
    - or_to_llm_1.txt
    - perfect_score_1.txt

- At the lead_time_* folder level (and small_batch folders), it removes:
    - batch_log_*.txt
    - batch_summary_*.json

It does NOT touch:
- Any train.csv / test.csv
- Any helper/check scripts or other code
"""

import os
from pathlib import Path

BASE = Path(__file__).parent
REAL_BASE = BASE / "real_trajectory"
SYN_BASE = BASE / "synthetic_trajectory"

INSTANCE_LOG_FILES = [
    "benchmark_results.json",
    "llm_1.txt",
    "llm_to_or_1.txt",
    "or_1.txt",
    "or_to_llm_1.txt",
    "perfect_score_1.txt",
]


def is_instance_dir(path: Path) -> bool:
    return path.is_dir() and (path / "test.csv").exists() and (path / "train.csv").exists()


def cleanup_instances(root: Path) -> int:
    removed = 0

    def recurse(p: Path):
        nonlocal removed
        if is_instance_dir(p):
            for name in INSTANCE_LOG_FILES:
                f = p / name
                if f.exists():
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
            return
        for child in p.iterdir():
            if child.is_dir():
                recurse(child)

    if root.exists():
        recurse(root)
    return removed


def cleanup_batch_logs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return 0
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        for fname in filenames:
            if fname.startswith("batch_log_") and fname.endswith(".txt"):
                f = p / fname
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
            elif fname.startswith("batch_summary_") and fname.endswith(".json"):
                f = p / fname
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def main():
    print("=" * 70)
    print("Cleaning old benchmark results and logs (for model switch)")
    print("=" * 70)

    # 1) Per-instance results/logs
    real_removed = cleanup_instances(REAL_BASE)
    syn_removed = cleanup_instances(SYN_BASE)
    print(f"Removed instance-level files: real={real_removed}, synthetic={syn_removed}")

    # 2) Batch logs/summaries for full datasets and small_batch helpers
    total_batch_removed = 0
    total_batch_removed += cleanup_batch_logs(REAL_BASE)
    total_batch_removed += cleanup_batch_logs(SYN_BASE)
    total_batch_removed += cleanup_batch_logs(BASE / "small_batch")
    print(f"Removed batch logs/summaries: {total_batch_removed}")

    print("=" * 70)
    print("Cleanup complete. You can now rerun run_batch_benchmark.py with the new model.")
    print("=" * 70)


if __name__ == "__main__":
    main()

