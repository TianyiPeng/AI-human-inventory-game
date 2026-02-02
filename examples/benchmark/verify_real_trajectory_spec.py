"""
Verify real_trajectory against BENCHMARK_SPECIFICATION.md:
1. Full spec compliance (counts, structure, lead times, P:H)
2. lead_time_stochastic: all test.csv have identical lead_time sequence
3. P:H ratio counts (1:1, 4:1, 19:1) across all instances
"""

import csv
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent / "real_trajectory"
SPEC_INSTANCES_TOTAL = 600
SPEC_ARTICLES_PER_LT = 200
EXPECTED_DATA_ROWS = (47, 48)  # spec says 48; source may have 47
VALID_PH = [(1, 1), (4, 1), (19, 1)]
VALID_PROFIT = {1, 4, 19}
VALID_HOLDING = {1}


def get_expected_lt(folder_name: str):
    if folder_name == "lead_time_0":
        return 0
    if folder_name == "lead_time_4":
        return 4
    if folder_name == "lead_time_stochastic":
        return None
    return None


def read_test_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def extract_lead_time_sequence(rows):
    """Return list of lead_time values (row by row)."""
    if len(rows) < 2:
        return []
    header = rows[0]
    # find lead_time column index
    lt_idx = None
    for i, col in enumerate(header):
        if "lead_time" in col.lower():
            lt_idx = i
            break
    if lt_idx is None:
        return None
    return [row[lt_idx].strip().lower() if len(row) > lt_idx else "" for row in rows[1:]]


def extract_profit_holding(rows):
    """Return (profit, holding_cost) from first data row. All rows should be same P:H."""
    if len(rows) < 2:
        return None, None
    header = rows[0]
    p_idx = h_idx = None
    for i, col in enumerate(header):
        if "profit" in col.lower():
            p_idx = i
        if "holding" in col.lower():
            h_idx = i
    if p_idx is None or h_idx is None:
        return None, None
    row = rows[1]
    if len(row) <= max(p_idx, h_idx):
        return None, None
    try:
        p = int(row[p_idx])
        h = int(row[h_idx])
        return p, h
    except ValueError:
        return None, None


def check_one_file(path: Path, expected_lt, errors: list) -> bool:
    """Validate one test.csv. Return True if ok."""
    try:
        rows = read_test_csv(path)
    except Exception as e:
        errors.append(f"Read error: {e}")
        return False
    if not rows:
        errors.append("Empty file")
        return False
    header = rows[0]
    data_rows = rows[1:]
    if len(header) != 6:
        errors.append(f"Header columns: {len(header)}, expected 6")
    if len(data_rows) not in EXPECTED_DATA_ROWS:
        errors.append(f"Data rows: {len(data_rows)}, expected one of {EXPECTED_DATA_ROWS}")
    for i, row in enumerate(data_rows):
        if len(row) != 6:
            errors.append(f"Row {i+2}: columns={len(row)}, expected 6")
            break
        try:
            lt_val = row[3].strip().lower()
            if lt_val == "inf":
                if expected_lt is not None:
                    errors.append(f"Row {i+2}: lead_time=inf in non-stochastic folder")
            else:
                lt_int = int(row[3])
                if expected_lt is not None and lt_int != expected_lt:
                    errors.append(f"Row {i+2}: lead_time={lt_int}, expected {expected_lt}")
        except ValueError:
            pass
        try:
            p, h = int(row[4]), int(row[5])
            if p not in VALID_PROFIT or h not in VALID_HOLDING:
                errors.append(f"Row {i+2}: profit={p}, holding={h} (expected P in {VALID_PROFIT}, H=1)")
        except ValueError:
            pass
    return len(errors) == 0


def main():
    print("=" * 70)
    print("REAL_TRAJECTORY vs BENCHMARK_SPECIFICATION.md")
    print("=" * 70)

    all_errors = {}
    ph_counter = Counter()
    lt_folders = ["lead_time_0", "lead_time_4", "lead_time_stochastic"]
    instance_count = 0
    per_lt_count = {}

    # 1) Spec compliance and P:H collection
    for lead_dir in lt_folders:
        lead_path = BASE / lead_dir
        if not lead_path.exists():
            print(f"  [MISSING] {lead_dir}/")
            continue
        expected_lt = get_expected_lt(lead_dir)
        dirs = [d for d in lead_path.iterdir() if d.is_dir()]
        per_lt_count[lead_dir] = len(dirs)
        for article_dir in sorted(dirs):
            test_csv = article_dir / "test.csv"
            rel = f"{lead_dir}/{article_dir.name}"
            if not test_csv.exists():
                all_errors[rel] = ["test.csv not found"]
                continue
            errs = []
            ok = check_one_file(test_csv, expected_lt, errs)
            if errs:
                all_errors[rel] = errs
            else:
                instance_count += 1
                rows = read_test_csv(test_csv)
                p, h = extract_profit_holding(rows)
                if p is not None and h is not None:
                    ph_counter[(p, h)] += 1

    # Report 1: counts
    print("\n1) INSTANCE COUNTS (spec: 600 total, 200 per lead-time folder)")
    print(f"   Total instances checked: {instance_count}")
    for lead_dir in lt_folders:
        n = per_lt_count.get(lead_dir, 0)
        status = "OK" if n == SPEC_ARTICLES_PER_LT else "MISMATCH"
        print(f"   {lead_dir}: {n} instances (expected {SPEC_ARTICLES_PER_LT}) [{status}]")
    if instance_count != SPEC_INSTANCES_TOTAL:
        print(f"   TOTAL: {instance_count} (expected {SPEC_INSTANCES_TOTAL}) [MISMATCH]")
    else:
        print(f"   TOTAL: {instance_count} [OK]")

    # Report 2: lead_time_stochastic same sequence
    print("\n2) LEAD_TIME_STOCHASTIC: same lead_time sequence in every test.csv?")
    stoch_path = BASE / "lead_time_stochastic"
    ref_sequence = None
    sequence_ok = True
    if stoch_path.exists():
        for article_dir in sorted(stoch_path.iterdir()):
            if not article_dir.is_dir():
                continue
            test_csv = article_dir / "test.csv"
            if not test_csv.exists():
                continue
            rows = read_test_csv(test_csv)
            seq = extract_lead_time_sequence(rows)
            if ref_sequence is None:
                ref_sequence = seq
            elif seq != ref_sequence:
                sequence_ok = False
                all_errors.setdefault(f"lead_time_stochastic/{article_dir.name}", []).append(
                    "Lead time sequence differs from other instances"
                )
        if ref_sequence is not None:
            print(f"   Reference sequence length: {len(ref_sequence)} periods")
            print(f"   First 10 values: {ref_sequence[:10]}")
        print(f"   All identical: {'YES [OK]' if sequence_ok else 'NO [MISMATCH]'}")
    else:
        print("   Folder not found.")

    # Report 3: P:H distribution
    print("\n3) P:H RATIO COUNTS (1:1, 4:1, 19:1)")
    for (p, h) in VALID_PH:
        label = f"  {p}:1"
        count = ph_counter.get((p, h), 0)
        print(f"   {label}: {count} test cases")
    total_ph = sum(ph_counter.values())
    print(f"   Total (with valid P:H): {total_ph}")

    # Errors summary
    print("\n4) SPEC VIOLATIONS (per-file)")
    if all_errors:
        print(f"   Files with errors: {len(all_errors)}")
        for rel, errs in list(all_errors.items())[:20]:
            print(f"   - {rel}: {errs[0]}")
        if len(all_errors) > 20:
            print(f"   ... and {len(all_errors) - 20} more.")
    else:
        print("   None.")

    print("=" * 70)
    if not all_errors and instance_count == SPEC_INSTANCES_TOTAL and sequence_ok:
        print("CONCLUSION: real_trajectory conforms to spec and is ready for experiments.")
    else:
        print("CONCLUSION: Fix the issues above before running experiments.")
    print("=" * 70)


if __name__ == "__main__":
    main()
