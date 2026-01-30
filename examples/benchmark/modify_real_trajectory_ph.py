"""
Modify real_trajectory to have random P:H ratios (uniform over low, med, high).

Each of the 600 instances (200 articles x 3 lead times) independently
gets a random P:H ratio from {1:1, 4:1, 19:1} via random.choice (uniform).

Only modifies the profit and holding_cost columns in test.csv.
"""

import random
from pathlib import Path

# P:H ratio configurations
PH_RATIOS = [
    {"name": "low", "profit": 1, "holding": 1},   # 1:1
    {"name": "med", "profit": 4, "holding": 1},   # 4:1
    {"name": "high", "profit": 19, "holding": 1},  # 19:1
]

BASE_DIR = Path(__file__).parent / "real_trajectory"
LEAD_TIMES = ["lead_time_0", "lead_time_4", "lead_time_stochastic"]

# Set seed for reproducibility
random.seed(42)


def modify_test_csv(test_csv_path: Path, profit: int, holding: int):
    """Modify profit and holding_cost columns in test.csv."""
    with open(test_csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return

    header = lines[0].strip().split(",")
    profit_col = None
    holding_col = None
    for i, col in enumerate(header):
        if "profit_" in col:
            profit_col = i
        if "holding_cost_" in col:
            holding_col = i

    if profit_col is None or holding_col is None:
        print(f"Warning: Could not find profit/holding columns in {test_csv_path}")
        return

    new_lines = [lines[0]]
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) > max(profit_col, holding_col):
            parts[profit_col] = str(profit)
            parts[holding_col] = str(holding)
        new_lines.append(",".join(parts) + "\n")

    with open(test_csv_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def main():
    stats = {"low": 0, "med": 0, "high": 0}
    total = 0

    for lead_time in LEAD_TIMES:
        lead_time_dir = BASE_DIR / lead_time
        if not lead_time_dir.exists():
            print(f"Warning: {lead_time_dir} does not exist")
            continue

        for article_dir in sorted(lead_time_dir.iterdir()):
            if not article_dir.is_dir():
                continue

            test_csv = article_dir / "test.csv"
            if not test_csv.exists():
                continue

            ph = random.choice(PH_RATIOS)  # uniform over {low, med, high}
            modify_test_csv(test_csv, ph["profit"], ph["holding"])
            stats[ph["name"]] += 1
            total += 1

            if total % 100 == 0:
                print(f"Processed {total} instances...")

    print("\n=== Summary ===")
    print(f"Total instances modified: {total}")
    for name, count in stats.items():
        print(f"  {name}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
