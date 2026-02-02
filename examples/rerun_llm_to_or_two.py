"""
Re-run llm_to_or only for the 2 incomplete instances, then update benchmark_results.json.
If benchmark_results.json is missing, build it from existing log files first.
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# The 2 instances missing llm_to_or
INSTANCES = [
    "examples/benchmark/synthetic_trajectory/lead_time_0/p07_seasonal/v4_multiplicative/r2_high",
    "examples/benchmark/synthetic_trajectory/lead_time_0/p07_seasonal/v4_multiplicative/r2_med",
]
SCRIPT = "examples/llm_to_or_csv_demo.py"
MODEL = "google/gemini-3-flash-preview"


def extract_reward(output: str):
    """Extract Total Reward from output."""
    m = re.search(r">>>\s*(?:Total Reward|Perfect Score)[^:]*:\s*\$(-?[\d,]+\.?\d*)\s*<<<", output)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"(?:Total Reward|Perfect Score)[^:]*:\s*\$(-?[\d,]+\.?\d*)", output)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def build_json_from_logs(instance_dir: Path) -> dict:
    """Build benchmark_results.json from existing log files."""
    def read_reward(name: str):
        p = instance_dir / f"{name}_1.txt"
        if p.exists():
            t = p.read_text(encoding="utf-8", errors="replace")
            return extract_reward(t)
        return None

    perfect = read_reward("perfect_score")
    or_r = read_reward("or")
    llm_r = read_reward("llm")
    or_to_llm_r = read_reward("or_to_llm")

    def mk(s, r):
        if r is None:
            return {"rewards": [], "mean": None, "std": None, "min": None, "max": None, "count": 0, "ratio_to_perfect": None}
        return {"rewards": [r], "mean": r, "std": 0.0, "min": r, "max": r, "count": 1,
                "ratio_to_perfect": r / perfect if perfect else None}

    ratios = {}
    if perfect:
        if or_r: ratios["or"] = or_r / perfect
        if llm_r: ratios["llm"] = llm_r / perfect
        if or_to_llm_r: ratios["or_to_llm"] = or_to_llm_r / perfect
        ratios["perfect_score"] = 1.0

    return {
        "instance_dir": str(instance_dir),
        "instance_name": instance_dir.name,
        "promised_lead_time": 0,
        "model": MODEL,
        "num_runs_llm": 1,
        "num_runs_deterministic": 1,
        "max_periods": None,
        "summary": {"perfect_score": perfect, "ratios": ratios, "warnings": None},
        "results": {
            "or": mk("or", or_r), "llm": mk("llm", llm_r),
            "llm_to_or": {"rewards": [], "mean": None, "std": None, "min": None, "max": None, "count": 0, "ratio_to_perfect": None},
            "or_to_llm": mk("or_to_llm", or_to_llm_r),
            "perfect_score": mk("perfect_score", perfect),
        },
        "errors": {},
    }


def run_llm_to_or(instance_path: str) -> tuple:
    """Run llm_to_or. Returns (reward, output)."""
    instance_dir = (BASE_DIR / instance_path).resolve()
    test_csv = instance_dir / "test.csv"
    train_csv = instance_dir / "train.csv"
    log_path = instance_dir / "llm_to_or_1.txt"

    promised = 0  # lead_time_0

    cmd = [
        sys.executable,
        str(BASE_DIR / SCRIPT),
        "--demand-file", str(test_csv),
        "--real-instance-train", str(train_csv),
        "--promised-lead-time", str(promised),
        "--model", MODEL,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, cwd=str(BASE_DIR))
    output = result.stdout + result.stderr

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)

    return extract_reward(output), output


def update_benchmark_json(instance_path: str, reward: float):
    """Update benchmark_results.json with llm_to_or result."""
    instance_dir = (BASE_DIR / instance_path).resolve()
    results_path = instance_dir / "benchmark_results.json"

    if not results_path.exists():
        print(f"  Building benchmark_results.json from existing logs...")
        data = build_json_from_logs(instance_dir)
    else:
        with open(results_path, "r") as f:
            data = json.load(f)

    perfect = data.get("summary", {}).get("perfect_score") or 0
    ratio = reward / perfect if perfect else None

    data["results"]["llm_to_or"] = {
        "rewards": [reward],
        "mean": reward,
        "std": 0.0,
        "min": reward,
        "max": reward,
        "count": 1,
        "ratio_to_perfect": ratio,
    }
    if data.get("summary", {}).get("ratios"):
        data["summary"]["ratios"]["llm_to_or"] = ratio
    if data.get("errors") and "llm_to_or" in data["errors"]:
        del data["errors"]["llm_to_or"]

    with open(instance_dir / "benchmark_results.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    print("Re-running llm_to_or for 2 instances...")
    for path in INSTANCES:
        name = path.split("/")[-2] + "/" + path.split("/")[-1]  # v4_multiplicative/r2_high
        print(f"\n[{name}]")
        reward, _ = run_llm_to_or(path)
        if reward is not None:
            print(f"  Reward: ${reward:.2f}")
            update_benchmark_json(path, reward)
            print(f"  Updated benchmark_results.json")
        else:
            print(f"  FAILED: could not extract reward")


if __name__ == "__main__":
    main()
