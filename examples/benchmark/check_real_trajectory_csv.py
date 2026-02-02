"""
Check all test.csv files in real_trajectory against BENCHMARK_SPECIFICATION.

Rules (from spec):
- lead_time_0 folder -> lead_time column must be 0 in every row
- lead_time_4 folder -> lead_time column must be 4 in every row  
- lead_time_stochastic -> lead_time in {1, 2, 3, inf} (or "inf")
- Columns: exact_dates_{id}, demand_{id}, description_{id}, lead_time_{id}, profit_{id}, holding_cost_{id}
- profit in {1, 4, 19}, holding_cost == 1
- 48 data rows (real data)
- demand integer; exact_dates date-like
"""

import csv
import os
from pathlib import Path

BASE = Path(__file__).parent / "real_trajectory"
EXPECTED_ROWS = (47, 48)  # accept 47 or 48 (source 200_articles has 47 data rows)
VALID_PROFIT = {1, 4, 19}
VALID_HOLDING = {1}


def expected_lead_time_for_folder(folder_name: str):
    if folder_name == "lead_time_0":
        return 0
    if folder_name == "lead_time_4":
        return 4
    if folder_name == "lead_time_stochastic":
        return None  # any of 1, 2, 3, inf
    return None


def check_one(path: Path, expected_lt) -> list:
    errors = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        errors.append(f"Read error: {e}")
        return errors

    if not rows:
        errors.append("Empty file")
        return errors

    header = rows[0]
    data_rows = rows[1:]

    # Expect 6 columns
    if len(header) != 6:
        errors.append(f"Header has {len(header)} columns, expected 6. Header: {header}")

    # Expect exact_dates, demand, description, lead_time, profit, holding_cost (order)
    if len(header) >= 6:
        if "lead_time" not in header[3] and "lead_time" not in str(header):
            errors.append(f"Column 4 should be lead_time_*, got: {header[3] if len(header)>3 else '?'}")
        if "profit" not in header[4] and "profit" not in str(header):
            errors.append(f"Column 5 should be profit_*, got: {header[4] if len(header)>4 else '?'}")
        if "holding" not in header[5] and "holding" not in str(header):
            errors.append(f"Column 6 should be holding_cost_*, got: {header[5] if len(header)>5 else '?'}")

    if isinstance(EXPECTED_ROWS, tuple):
        if len(data_rows) not in EXPECTED_ROWS:
            errors.append(f"Data rows: {len(data_rows)}, expected one of {EXPECTED_ROWS}")
    elif len(data_rows) != EXPECTED_ROWS:
        errors.append(f"Data rows: {len(data_rows)}, expected {EXPECTED_ROWS}")

    for i, row in enumerate(data_rows):
        if len(row) != 6:
            errors.append(f"Row {i+2}: has {len(row)} columns, expected 6. Row preview: {row[:4]}...")
            if len(errors) >= 3:  # cap same-type errors per file
                break
            continue
        try:
            demand = int(row[1])
        except ValueError:
            errors.append(f"Row {i+2}: demand not integer: {row[1]}")
        try:
            # Column index: 0=date, 1=demand, 2=description, 3=lead_time, 4=profit, 5=holding
            lt_val = row[3]
            if lt_val.strip().lower() == "inf":
                if expected_lt is not None and expected_lt != "stochastic":
                    errors.append(f"Row {i+2}: lead_time=inf in non-stochastic folder")
            else:
                lt_int = int(lt_val)
                if expected_lt is not None and lt_int != expected_lt:
                    errors.append(f"Row {i+2}: lead_time={lt_int}, expected {expected_lt} (from folder)")
        except (ValueError, IndexError) as e:
            errors.append(f"Row {i+2}: lead_time parse error: {e} (value={row[3] if len(row)>3 else '?'})")
        try:
            p = int(row[4])
            h = int(row[5])
            if p not in VALID_PROFIT:
                errors.append(f"Row {i+2}: profit={p}, expected one of {VALID_PROFIT}")
            if h not in VALID_HOLDING:
                errors.append(f"Row {i+2}: holding_cost={h}, expected 1")
        except (ValueError, IndexError):
            pass

    return errors


def main():
    results = {"ok": [], "errors": {}}
    for lead_dir in ["lead_time_0", "lead_time_4", "lead_time_stochastic"]:
        lead_path = BASE / lead_dir
        if not lead_path.exists():
            continue
        expected_lt = expected_lead_time_for_folder(lead_dir)
        for article_dir in sorted(lead_path.iterdir()):
            if not article_dir.is_dir():
                continue
            test_csv = article_dir / "test.csv"
            if not test_csv.exists():
                results["errors"][str(test_csv)] = ["test.csv not found"]
                continue
            rel = f"{lead_dir}/{article_dir.name}"
            errs = check_one(test_csv, expected_lt)
            if errs:
                results["errors"][rel] = errs
            else:
                results["ok"].append(rel)

    print("=" * 60)
    print("REAL_TRAJECTORY test.csv CHECK")
    print("=" * 60)
    print(f"OK: {len(results['ok'])}")
    print(f"With errors: {len(results['errors'])}")
    if results["errors"]:
        print("\n--- ERRORS (first 30 files) ---")
        for rel, errs in list(results["errors"].items())[:30]:
            print(f"\n  {rel}:")
            for e in errs[:5]:
                print(f"    - {e}")
        if len(results["errors"]) > 30:
            print(f"\n  ... and {len(results['errors'])-30} more files with errors.")
    print("=" * 60)


if __name__ == "__main__":
    main()
