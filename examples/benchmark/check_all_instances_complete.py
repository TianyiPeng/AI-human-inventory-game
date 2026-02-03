"""
Strict check: 600 (real_trajectory) + 720 (synthetic_trajectory) instances are fully benchmarked.

For each instance:
1. benchmark_results.json must exist.
2. All 5 methods must have valid results: or, llm, llm_to_or, or_to_llm, perfect_score
   (rewards list non-empty, mean is a number).
3. promised_lead_time in JSON must match folder:
   - lead_time_0 -> 0
   - lead_time_4 -> 4
   - lead_time_stochastic -> 2
"""

import json
import os
from pathlib import Path
from collections import defaultdict

REQUIRED_METHODS = ["or", "llm", "llm_to_or", "or_to_llm", "perfect_score"]
FOLDER_TO_PROMISED_LT = {
    "lead_time_0": 0,
    "lead_time_4": 4,
    "lead_time_stochastic": 2,
}
EXPECTED_REAL = 600
EXPECTED_SYNTHETIC = 720
BENCHMARK_DIR = Path(__file__).parent
REAL_BASE = BENCHMARK_DIR / "real_trajectory"
SYNTHETIC_BASE = BENCHMARK_DIR / "synthetic_trajectory"


def find_instances(base_path: Path) -> list:
    """Return list of (instance_path_str, expected_promised_lt)."""
    out = []
    if not base_path.is_dir():
        return out
    for lt_folder in ["lead_time_0", "lead_time_4", "lead_time_stochastic"]:
        lt_path = base_path / lt_folder
        expected_lt = FOLDER_TO_PROMISED_LT[lt_folder]
        if not lt_path.is_dir():
            continue
        for item in sorted(lt_path.iterdir()):
            if not item.is_dir():
                continue
            # Real: instance = lt_path / article_id; Synthetic: instance = lt_path / pattern / variant / realization
            instances_in_this_branch = _collect_instances_under(item, expected_lt)
            out.extend(instances_in_this_branch)
    return out


def _collect_instances_under(path: Path, expected_lt: int) -> list:
    """If path has test.csv+train.csv it's an instance; else recurse."""
    if (path / "test.csv").exists() and (path / "train.csv").exists():
        return [(str(path), expected_lt)]
    out = []
    for item in sorted(path.iterdir()):
        if item.is_dir():
            out.extend(_collect_instances_under(item, expected_lt))
    return out


def is_method_valid(results: dict, method: str) -> bool:
    if method not in results:
        return False
    r = results[method]
    if not isinstance(r, dict):
        return False
    rewards = r.get("rewards")
    if not isinstance(rewards, list) or len(rewards) == 0:
        return False
    mean = r.get("mean")
    if mean is None and (rewards is None or len(rewards) == 0):
        return False
    try:
        if mean is not None:
            float(mean)
    except (TypeError, ValueError):
        return False
    return True


def check_one(instance_path: str, expected_promised_lt: int) -> tuple:
    """
    Returns (ok: bool, issues: list of str).
    """
    issues = []
    json_path = Path(instance_path) / "benchmark_results.json"
    if not json_path.exists():
        return False, ["benchmark_results.json missing"]
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"benchmark_results.json read error: {e}"]
    results = data.get("results", {})
    for method in REQUIRED_METHODS:
        if not is_method_valid(results, method):
            issues.append(f"method '{method}' missing or invalid (rewards empty or mean null)")
    promised = data.get("promised_lead_time")
    if promised is None:
        issues.append("promised_lead_time missing in JSON")
    else:
        try:
            promised_int = int(promised)
            if promised_int != expected_promised_lt:
                issues.append(
                    f"promised_lead_time={promised_int} does not match folder (expected {expected_promised_lt})"
                )
        except (TypeError, ValueError):
            issues.append(f"promised_lead_time not an integer: {promised}")
    return len(issues) == 0, issues


def main():
    print("=" * 70)
    print("BENCHMARK COMPLETENESS CHECK (600 real + 720 synthetic)")
    print("=" * 70)

    real_instances = find_instances(REAL_BASE)
    synthetic_instances = find_instances(SYNTHETIC_BASE)

    print(f"\nReal trajectory: found {len(real_instances)} instances (expected {EXPECTED_REAL})")
    print(f"Synthetic trajectory: found {len(synthetic_instances)} instances (expected {EXPECTED_SYNTHETIC})")

    all_ok = True
    if len(real_instances) != EXPECTED_REAL:
        print(f"  [MISMATCH] Real instance count: expected {EXPECTED_REAL}")
        all_ok = False
    if len(synthetic_instances) != EXPECTED_SYNTHETIC:
        print(f"  [MISMATCH] Synthetic instance count: expected {EXPECTED_SYNTHETIC}")
        all_ok = False

    # Check real
    real_missing_json = []
    real_invalid = []  # (path, issues)
    real_ok_count = 0
    for path_str, expected_lt in real_instances:
        ok, issues = check_one(path_str, expected_lt)
        if ok:
            real_ok_count += 1
        else:
            rel = str(Path(path_str).relative_to(REAL_BASE)) if path_str.startswith(str(REAL_BASE)) else path_str
            if any("missing" in i.lower() for i in issues) and not (Path(path_str) / "benchmark_results.json").exists():
                real_missing_json.append(rel)
            else:
                real_invalid.append((rel, issues))

    # Check synthetic
    syn_missing_json = []
    syn_invalid = []
    syn_ok_count = 0
    for path_str, expected_lt in synthetic_instances:
        ok, issues = check_one(path_str, expected_lt)
        if ok:
            syn_ok_count += 1
        else:
            rel = str(Path(path_str).relative_to(SYNTHETIC_BASE)) if path_str.startswith(str(SYNTHETIC_BASE)) else path_str
            if any("missing" in i.lower() for i in issues) and not (Path(path_str) / "benchmark_results.json").exists():
                syn_missing_json.append(rel)
            else:
                syn_invalid.append((rel, issues))

    # Report real
    print("\n--- REAL_TRAJECTORY ---")
    print(f"  Complete (all 5 methods valid + promised_lead_time match): {real_ok_count}/{len(real_instances)}")
    if real_missing_json:
        print(f"  Missing benchmark_results.json: {len(real_missing_json)}")
        for p in real_missing_json[:15]:
            print(f"    - {p}")
        if len(real_missing_json) > 15:
            print(f"    ... and {len(real_missing_json) - 15} more")
        all_ok = False
    if real_invalid:
        print(f"  Incomplete or invalid (method/promised_lead_time): {len(real_invalid)}")
        for rel, issues in real_invalid[:15]:
            print(f"    - {rel}")
            for i in issues[:5]:
                print(f"      {i}")
        if len(real_invalid) > 15:
            print(f"    ... and {len(real_invalid) - 15} more")
        all_ok = False

    # Report synthetic
    print("\n--- SYNTHETIC_TRAJECTORY ---")
    print(f"  Complete (all 5 methods valid + promised_lead_time match): {syn_ok_count}/{len(synthetic_instances)}")
    if syn_missing_json:
        print(f"  Missing benchmark_results.json: {len(syn_missing_json)}")
        for p in syn_missing_json[:15]:
            print(f"    - {p}")
        if len(syn_missing_json) > 15:
            print(f"    ... and {len(syn_missing_json) - 15} more")
        all_ok = False
    if syn_invalid:
        print(f"  Incomplete or invalid (method/promised_lead_time): {len(syn_invalid)}")
        for rel, issues in syn_invalid[:15]:
            print(f"    - {rel}")
            for i in issues[:5]:
                print(f"      {i}")
        if len(syn_invalid) > 15:
            print(f"    ... and {len(syn_invalid) - 15} more")
        all_ok = False

    # Summary
    total_complete = real_ok_count + syn_ok_count
    total_expected = EXPECTED_REAL + EXPECTED_SYNTHETIC
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_complete}/{total_expected} instances fully complete (all 5 methods valid, promised_lead_time matches folder)")
    if all_ok and total_complete == total_expected:
        print("CONCLUSION: All 600 + 720 instances are tested and valid.")
    else:
        print("CONCLUSION: There are missing or invalid cases. Fix above and re-run benchmarks as needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
