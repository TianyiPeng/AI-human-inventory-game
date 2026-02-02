"""
Verify synthetic_trajectory against BENCHMARK_SPECIFICATION.md.

Expected (from spec):
- 3 lead-time folders: lead_time_0, lead_time_4, lead_time_stochastic
- Under each: 10 patterns (p01_* ... p10_*), 4 variants each, 6 realizations each
  -> 10 x 4 x 6 = 240 instances per lead-time, 720 total
- Realization names: r1_low, r1_med, r1_high, r2_low, r2_med, r2_high
- train.csv: 2 columns (exact_dates_*, demand_*), 5 data rows
- test.csv: 5 columns (exact_dates, demand, lead_time, profit, holding_cost), 50 data rows
- lead_time: 0 in lead_time_0, 4 in lead_time_4, values in {1,2,3,inf} in lead_time_stochastic
- P:H: r*_low -> (1,1), r*_med -> (4,1), r*_high -> (19,1)
- lead_time_stochastic: same lead_time sequence in every instance (comparability)
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).parent / "synthetic_trajectory"
EXPECTED_LT_FOLDERS = ["lead_time_0", "lead_time_4", "lead_time_stochastic"]
EXPECTED_PATTERNS = 10
EXPECTED_VARIANTS_PER_PATTERN = 4
EXPECTED_REALIZATIONS = ["r1_low", "r1_med", "r1_high", "r2_low", "r2_med", "r2_high"]
EXPECTED_TRAIN_ROWS = 5
EXPECTED_TEST_ROWS = 50
EXPECTED_TEST_COLS = 5
EXPECTED_TRAIN_COLS = 2
VALID_PROFIT = {1, 4, 19}
VALID_HOLDING = {1}
# P:H by realization name
EXPECTED_PH = {"r1_low": (1, 1), "r1_med": (4, 1), "r1_high": (19, 1), "r2_low": (1, 1), "r2_med": (4, 1), "r2_high": (19, 1)}


def read_csv_rows(path: Path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            return list(reader)
    except Exception as e:
        return None


def get_expected_lt(folder_name: str):
    if folder_name == "lead_time_0":
        return 0
    if folder_name == "lead_time_4":
        return 4
    if folder_name == "lead_time_stochastic":
        return None  # 1, 2, 3, inf
    return None


def check_train_csv(path: Path, errors: list) -> bool:
    rows = read_csv_rows(path)
    if rows is None:
        errors.append("train.csv: read error")
        return False
    if len(rows) < 2:
        errors.append(f"train.csv: too few rows ({len(rows)})")
        return False
    header, data = rows[0], rows[1:]
    if len(header) != EXPECTED_TRAIN_COLS:
        errors.append(f"train.csv: header has {len(header)} columns, expected {EXPECTED_TRAIN_COLS}")
    if len(data) != EXPECTED_TRAIN_ROWS:
        errors.append(f"train.csv: {len(data)} data rows, expected {EXPECTED_TRAIN_ROWS}")
    if "exact_dates" not in str(header[0]).lower() or "demand" not in str(header[1]).lower():
        errors.append(f"train.csv: expected exact_dates_*, demand_* columns; got {header[:2]}")
    for i, row in enumerate(data):
        if len(row) != 2:
            errors.append(f"train.csv row {i+2}: expected 2 columns, got {len(row)}")
            break
        try:
            int(row[1])
        except ValueError:
            errors.append(f"train.csv row {i+2}: demand not integer ({row[1]})")
            break
    return len(errors) == 0


def check_test_csv(path: Path, expected_lt, realization_name: str, errors: list) -> bool:
    rows = read_csv_rows(path)
    if rows is None:
        errors.append("test.csv: read error")
        return False
    if not rows:
        errors.append("test.csv: empty")
        return False
    header, data = rows[0], rows[1:]
    if len(header) != EXPECTED_TEST_COLS:
        errors.append(f"test.csv: header has {len(header)} columns, expected {EXPECTED_TEST_COLS}")
    if len(data) != EXPECTED_TEST_ROWS:
        errors.append(f"test.csv: {len(data)} data rows, expected {EXPECTED_TEST_ROWS}")
    # Column roles: 0=exact_dates, 1=demand, 2=lead_time, 3=profit, 4=holding_cost
    if "lead_time" not in str(header[2]).lower():
        errors.append(f"test.csv: column 3 should be lead_time_*, got {header[2] if len(header) > 2 else '?'}")
    if "profit" not in str(header[3]).lower():
        errors.append(f"test.csv: column 4 should be profit_*, got {header[3] if len(header) > 3 else '?'}")
    if "holding" not in str(header[4]).lower():
        errors.append(f"test.csv: column 5 should be holding_cost_*, got {header[4] if len(header) > 4 else '?'}")
    expected_p, expected_h = EXPECTED_PH.get(realization_name, (None, None))
    ph_checked = False
    for i, row in enumerate(data):
        if len(row) != EXPECTED_TEST_COLS:
            errors.append(f"test.csv row {i+2}: expected {EXPECTED_TEST_COLS} columns, got {len(row)}")
            if len(errors) >= 5:
                break
            continue
        try:
            demand = int(row[1])
        except ValueError:
            errors.append(f"test.csv row {i+2}: demand not integer ({row[1]})")
        lt_val = row[2].strip().lower()
        if lt_val == "inf":
            if expected_lt is not None:
                errors.append(f"test.csv row {i+2}: lead_time=inf in non-stochastic folder")
        else:
            try:
                lt_int = int(lt_val)
                if expected_lt is not None and lt_int != expected_lt:
                    errors.append(f"test.csv row {i+2}: lead_time={lt_int}, expected {expected_lt}")
            except ValueError:
                errors.append(f"test.csv row {i+2}: lead_time not int or inf ({row[2]})")
        try:
            p, h = int(row[3]), int(row[4])
            if p not in VALID_PROFIT or h not in VALID_HOLDING:
                errors.append(f"test.csv row {i+2}: profit={p}, holding={h} (expected P in {VALID_PROFIT}, H=1)")
            if not ph_checked and expected_p is not None:
                if (p, h) != (expected_p, expected_h):
                    errors.append(f"test.csv: P:H ({p},{h}) does not match {realization_name} (expected {expected_p}:{expected_h})")
                ph_checked = True
        except ValueError:
            pass
        if len(errors) >= 5:
            break
    return len(errors) == 0


def extract_lead_time_sequence(rows):
    if not rows or len(rows) < 2:
        return None
    return [row[2].strip().lower() if len(row) > 2 else "" for row in rows[1:]]


def main():
    print("=" * 70)
    print("SYNTHETIC_TRAJECTORY vs BENCHMARK_SPECIFICATION.md")
    print("=" * 70)

    all_errors = defaultdict(list)
    total_instances = 0
    per_lt_count = {}
    stoch_sequences = []  # (instance_path, sequence) for lead_time_stochastic
    pattern_names = []

    for lt_folder in EXPECTED_LT_FOLDERS:
        lt_path = BASE / lt_folder
        if not lt_path.exists():
            all_errors["structure"].append(f"Missing folder: {lt_folder}")
            continue
        expected_lt = get_expected_lt(lt_folder)
        pattern_dirs = sorted([d for d in lt_path.iterdir() if d.is_dir()])
        # Expect 10 patterns; names should start with p01, p02, ... p10
        if not pattern_names:
            pattern_names = [d.name for d in pattern_dirs]
        if len(pattern_dirs) != EXPECTED_PATTERNS:
            all_errors["structure"].append(
                f"{lt_folder}: expected {EXPECTED_PATTERNS} pattern dirs, got {len(pattern_dirs)}"
            )
        lt_instances = 0
        for pattern_dir in pattern_dirs:
            variant_dirs = sorted([d for d in pattern_dir.iterdir() if d.is_dir()])
            if len(variant_dirs) != EXPECTED_VARIANTS_PER_PATTERN:
                all_errors["structure"].append(
                    f"{lt_folder}/{pattern_dir.name}: expected {EXPECTED_VARIANTS_PER_PATTERN} variants, got {len(variant_dirs)}"
                )
            for variant_dir in variant_dirs:
                real_dirs = sorted([d.name for d in variant_dir.iterdir() if d.is_dir()])
                if set(real_dirs) != set(EXPECTED_REALIZATIONS):
                    missing = set(EXPECTED_REALIZATIONS) - set(real_dirs)
                    extra = set(real_dirs) - set(EXPECTED_REALIZATIONS)
                    if missing or extra:
                        all_errors["structure"].append(
                            f"{lt_folder}/{pattern_dir.name}/{variant_dir.name}: "
                            f"realizations mismatch (missing: {missing or 'none'}, extra: {extra or 'none'})"
                        )
                for rname in EXPECTED_REALIZATIONS:
                    inst_dir = variant_dir / rname
                    if not inst_dir.is_dir():
                        all_errors["structure"].append(f"Missing: {inst_dir.relative_to(BASE)}")
                        continue
                    train_path = inst_dir / "train.csv"
                    test_path = inst_dir / "test.csv"
                    rel = f"{lt_folder}/{pattern_dir.name}/{variant_dir.name}/{rname}"
                    errs = []
                    if not train_path.exists():
                        errs.append("train.csv missing")
                    else:
                        check_train_csv(train_path, errs)
                    if not test_path.exists():
                        errs.append("test.csv missing")
                    else:
                        check_test_csv(test_path, expected_lt, rname, errs)
                    if errs:
                        all_errors[rel] = errs
                    else:
                        lt_instances += 1
                        total_instances += 1
                        if lt_folder == "lead_time_stochastic":
                            rows = read_csv_rows(test_path)
                            if rows:
                                seq = extract_lead_time_sequence(rows)
                                if seq is not None:
                                    stoch_sequences.append((rel, seq))
        per_lt_count[lt_folder] = lt_instances

    # Check lead_time_stochastic: same sequence everywhere
    same_stoch_sequence = True
    if stoch_sequences:
        ref_seq = stoch_sequences[0][1]
        for rel, seq in stoch_sequences[1:]:
            if seq != ref_seq:
                same_stoch_sequence = False
                all_errors[rel].append("Lead time sequence differs from other stochastic instances")
        print("\n2) LEAD_TIME_STOCHASTIC: same lead_time sequence in every test.csv?")
        print(f"   Reference sequence length: {len(ref_seq)} periods")
        print(f"   First 10 values: {ref_seq[:10]}")
        print(f"   All identical: {'YES [OK]' if same_stoch_sequence else 'NO [MISMATCH]'}")
    else:
        print("\n2) LEAD_TIME_STOCHASTIC: no instances checked (folder missing or empty).")

    # Report
    print("\n1) STRUCTURE & COUNTS (spec: 720 total, 240 per lead-time folder)")
    print(f"   Pattern folder names: {pattern_names}")
    print(f"   Total instances checked: {total_instances}")
    for lt_folder in EXPECTED_LT_FOLDERS:
        n = per_lt_count.get(lt_folder, 0)
        status = "OK" if n == 240 else "MISMATCH"
        print(f"   {lt_folder}: {n} instances (expected 240) [{status}]")
    if total_instances != 720:
        print(f"   TOTAL: {total_instances} (expected 720) [MISMATCH]")
    else:
        print(f"   TOTAL: {total_instances} [OK]")

    print("\n3) SPEC VIOLATIONS (per-file or structure)")
    structure_errs = all_errors.pop("structure", [])
    if structure_errs:
        print("   Structure:")
        for e in structure_errs[:15]:
            print(f"     - {e}")
        if len(structure_errs) > 15:
            print(f"     ... and {len(structure_errs) - 15} more.")
    if all_errors:
        print(f"   Files with errors: {len(all_errors)}")
        for rel, errs in list(all_errors.items())[:10]:
            print(f"     - {rel}: {errs[0]}")
        if len(all_errors) > 10:
            print(f"     ... and {len(all_errors) - 10} more.")
    if not structure_errs and not all_errors:
        print("   None.")

    print("=" * 70)
    if not structure_errs and not all_errors and total_instances == 720 and same_stoch_sequence:
        print("CONCLUSION: synthetic_trajectory conforms to spec and is ready for experiments.")
    else:
        print("CONCLUSION: Fix the issues above before relying on synthetic_trajectory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
