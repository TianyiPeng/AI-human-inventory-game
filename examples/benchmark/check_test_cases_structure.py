"""
Check test cases structure matches specification:

Real data:
- 200 articles × 3 lead times = 600 instances
- Each instance randomly assigned P:H ratio (1:1, 4:1, 19:1)
- Check: distribution of P:H ratios across 600 instances

Synthetic data:
- 40 patterns (10 patterns × 4 variants) × 3 lead times × 3 P:H ratios × 2 draws = 720 instances
- Check: structure matches this formula
"""

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

BENCHMARK_DIR = Path(__file__).parent
REAL_BASE = BENCHMARK_DIR / "real_trajectory"
SYNTHETIC_BASE = BENCHMARK_DIR / "synthetic_trajectory"

# P:H ratios
PH_RATIOS = [(1, 1), (4, 1), (19, 1)]
PH_LABELS = {(1, 1): "1:1", (4, 1): "4:1", (19, 1): "19:1"}


def get_ph_from_test_csv(test_csv_path: Path) -> tuple:
    """Extract (profit, holding_cost) from first data row of test.csv."""
    try:
        with open(test_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) < 5:
                return None, None
            # Find profit and holding_cost column indices
            # Columns: exact_dates, demand, [description], lead_time, profit, holding_cost
            # For real: profit_{id}, holding_cost_{id}
            # For synthetic: profit_chips(Regular), holding_cost_chips(Regular)
            p_idx = h_idx = None
            for i, col in enumerate(header):
                col_lower = col.lower()
                if "profit" in col_lower and p_idx is None:
                    p_idx = i
                if "holding" in col_lower and h_idx is None:
                    h_idx = i
            if p_idx is None or h_idx is None:
                return None, None
            # Read first data row
            row = next(reader, None)
            if row is None or len(row) <= max(p_idx, h_idx):
                return None, None
            try:
                p = int(row[p_idx])
                h = int(row[h_idx])
                return p, h
            except ValueError:
                return None, None
    except Exception:
        return None, None


def check_real_data():
    """Check real_trajectory: 200 × 3 = 600, P:H distribution."""
    print("=" * 70)
    print("REAL DATA STRUCTURE CHECK")
    print("=" * 70)

    ph_counter = Counter()
    per_lt_count = Counter()
    total_instances = 0
    missing_test_csv = []

    for lt_folder in ["lead_time_0", "lead_time_4", "lead_time_stochastic"]:
        lt_path = REAL_BASE / lt_folder
        if not lt_path.exists():
            print(f"  [MISSING] {lt_folder}/")
            continue
        for article_dir in sorted(lt_path.iterdir()):
            if not article_dir.is_dir():
                continue
            total_instances += 1
            per_lt_count[lt_folder] += 1
            test_csv = article_dir / "test.csv"
            if not test_csv.exists():
                missing_test_csv.append(f"{lt_folder}/{article_dir.name}")
                continue
            p, h = get_ph_from_test_csv(test_csv)
            if p is not None and h is not None:
                ph_counter[(p, h)] += 1

    print(f"\nTotal instances: {total_instances} (expected 600)")
    print(f"  lead_time_0: {per_lt_count['lead_time_0']} (expected 200)")
    print(f"  lead_time_4: {per_lt_count['lead_time_4']} (expected 200)")
    print(f"  lead_time_stochastic: {per_lt_count['lead_time_stochastic']} (expected 200)")

    if missing_test_csv:
        print(f"\n  [WARNING] Missing test.csv: {len(missing_test_csv)} instances")
        for p in missing_test_csv[:10]:
            print(f"    - {p}")

    print(f"\nP:H Ratio Distribution (from test.csv):")
    total_ph = sum(ph_counter.values())
    for ph in PH_RATIOS:
        label = PH_LABELS[ph]
        count = ph_counter.get(ph, 0)
        pct = (count / total_ph * 100) if total_ph > 0 else 0
        print(f"  {label}: {count} instances ({pct:.1f}%)")
    print(f"  Total with valid P:H: {total_ph}")

    # Check if distribution is roughly balanced (each should be ~200 if random)
    expected_per_ph = total_instances / 3
    print(f"\nExpected per P:H ratio (if uniform): ~{expected_per_ph:.0f}")
    balanced = all(abs(ph_counter.get(ph, 0) - expected_per_ph) < expected_per_ph * 0.3 for ph in PH_RATIOS)
    if balanced:
        print("  [OK] Distribution appears balanced (within 30% of expected)")
    else:
        print("  [NOTE] Distribution may not be uniform (this is OK if random seed was used)")

    return total_instances == 600 and per_lt_count['lead_time_0'] == 200 and per_lt_count['lead_time_4'] == 200 and per_lt_count['lead_time_stochastic'] == 200


def check_synthetic_data():
    """Check synthetic_trajectory: 40 patterns × 3 lead times × 3 P:H × 2 draws = 720."""
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA STRUCTURE CHECK")
    print("=" * 70)

    # Expected: 10 patterns × 4 variants = 40 distributions
    # Each distribution: 3 lead times × 3 P:H × 2 draws = 18 instances
    # Total: 40 × 18 = 720

    pattern_names = set()
    variant_names_per_pattern = defaultdict(set)  # pattern -> set of variant names
    realization_counts_per_variant = defaultdict(int)  # (pattern, variant) -> count across all lead_times
    per_lt_count = Counter()
    ph_counts = Counter()  # Count P:H ratios
    total_instances = 0

    EXPECTED_REALIZATIONS = ["r1_low", "r1_med", "r1_high", "r2_low", "r2_med", "r2_high"]
    EXPECTED_PH = {
        "r1_low": (1, 1), "r1_med": (4, 1), "r1_high": (19, 1),
        "r2_low": (1, 1), "r2_med": (4, 1), "r2_high": (19, 1),
    }

    for lt_folder in ["lead_time_0", "lead_time_4", "lead_time_stochastic"]:
        lt_path = SYNTHETIC_BASE / lt_folder
        if not lt_path.exists():
            print(f"  [MISSING] {lt_folder}/")
            continue
        for pattern_dir in sorted(lt_path.iterdir()):
            if not pattern_dir.is_dir() or not pattern_dir.name.startswith("p"):
                continue
            pattern_name = pattern_dir.name
            pattern_names.add(pattern_name)
            for variant_dir in sorted(pattern_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                variant_name = variant_dir.name
                variant_names_per_pattern[pattern_name].add(variant_name)
                key = (pattern_name, variant_name)
                for rname in EXPECTED_REALIZATIONS:
                    inst_dir = variant_dir / rname
                    if (inst_dir / "test.csv").exists() and (inst_dir / "train.csv").exists():
                        realization_counts_per_variant[key] += 1
                        total_instances += 1
                        per_lt_count[lt_folder] += 1
                        # Get P:H from test.csv
                        test_csv = inst_dir / "test.csv"
                        p, h = get_ph_from_test_csv(test_csv)
                        if p is not None and h is not None:
                            ph_counts[(p, h)] += 1
                            # Verify it matches realization name
                            expected_ph = EXPECTED_PH.get(rname)
                            if expected_ph and (p, h) != expected_ph:
                                print(f"  [WARNING] {lt_folder}/{pattern_name}/{variant_name}/{rname}: P:H ({p},{h}) != expected {expected_ph}")

    print(f"\nPatterns found: {len(pattern_names)} (expected 10)")
    print(f"  Pattern names: {sorted(pattern_names)}")

    # Check variants per pattern (should be 4 unique variants, each appears in all 3 lead_time folders)
    print(f"\nVariants per pattern (unique variants, should be 4):")
    all_have_4 = True
    for pname in sorted(pattern_names):
        vcount = len(variant_names_per_pattern[pname])
        if vcount != 4:
            print(f"  {pname}: {vcount} variants [MISMATCH - expected 4]")
            all_have_4 = False
        else:
            print(f"  {pname}: {vcount} variants [OK]")
    total_distributions = sum(len(v) for v in variant_names_per_pattern.values())
    print(f"  Total unique distributions: {total_distributions} (expected 40)")

    # Check realizations per variant (should be 6 per variant, but counted across all 3 lead_time folders = 18)
    print(f"\nRealizations per variant (across all 3 lead_time folders, should be 18 = 6 × 3):")
    all_have_18 = True
    for (pname, vname), rcount in sorted(realization_counts_per_variant.items()):
        if rcount != 18:
            print(f"  {pname}/{vname}: {rcount} realizations [MISMATCH - expected 18]")
            all_have_18 = False
    if all_have_18:
        print("  [OK] All variants have 18 realizations (6 per lead_time × 3 lead_times)")

    print(f"\nInstances per lead time:")
    print(f"  lead_time_0: {per_lt_count['lead_time_0']} (expected 240)")
    print(f"  lead_time_4: {per_lt_count['lead_time_4']} (expected 240)")
    print(f"  lead_time_stochastic: {per_lt_count['lead_time_stochastic']} (expected 240)")
    print(f"  Total: {total_instances} (expected 720)")

    print(f"\nP:H Ratio Distribution (from test.csv):")
    total_ph = sum(ph_counts.values())
    for ph in PH_RATIOS:
        label = PH_LABELS[ph]
        count = ph_counts.get(ph, 0)
        pct = (count / total_ph * 100) if total_ph > 0 else 0
        print(f"  {label}: {count} instances ({pct:.1f}%)")
    print(f"  Expected per P:H: 240 (720 / 3)")
    print(f"  Total with valid P:H: {total_ph}")

    # Verify formula: 40 × 3 × 3 × 2 = 720
    expected = 40 * 3 * 3 * 2
    matches = (
        len(pattern_names) == 10
        and total_distributions == 40
        and all_have_4
        and all_have_18
        and total_instances == expected
        and per_lt_count['lead_time_0'] == 240
        and per_lt_count['lead_time_4'] == 240
        and per_lt_count['lead_time_stochastic'] == 240
        and total_ph == 720
    )
    return matches


def main():
    print("=" * 70)
    print("TEST CASES STRUCTURE VERIFICATION")
    print("=" * 70)
    print("\nSpecification:")
    print("  Real: 200 articles × 3 lead times = 600 instances (random P:H per instance)")
    print("  Synthetic: 40 patterns × 3 lead times × 3 P:H × 2 draws = 720 instances")
    print("=" * 70)

    real_ok = check_real_data()
    synthetic_ok = check_synthetic_data()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Real data structure: {'[OK]' if real_ok else '[MISMATCH]'}")
    print(f"Synthetic data structure: {'[OK]' if synthetic_ok else '[MISMATCH]'}")
    if real_ok and synthetic_ok:
        print("\nCONCLUSION: All test cases match the specification.")
    else:
        print("\nCONCLUSION: Some mismatches found. See details above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
