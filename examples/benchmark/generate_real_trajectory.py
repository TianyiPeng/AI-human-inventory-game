"""
Regenerate real_trajectory from 200_articles per BENCHMARK_SPECIFICATION.md.

Steps:
1. Copy 200_articles into three lead-time folders (lead_time_0, lead_time_4, lead_time_stochastic).
2. Overwrite lead_time column in each test.csv according to folder (0, 4, or stochastic sequence).
3. Assign random P:H ratio to each of the 600 instances (seed=42): {1:1, 4:1, 19:1}.

All CSV writing uses proper quoting so description (with commas) is safe.
"""

import csv
import random
import shutil
from pathlib import Path

SOURCE = Path(__file__).parent / "200_articles"
OUT = Path(__file__).parent / "real_trajectory"
LEAD_FOLDERS = ["lead_time_0", "lead_time_4", "lead_time_stochastic"]
PH_RATIOS = [(1, 1), (4, 1), (19, 1)]  # (profit, holding_cost)
STOCHASTIC_VALUES = [1, 2, 3, "inf"]
SEED = 42


def get_article_dirs():
    """List article directories (those with test.csv and train.csv)."""
    articles = []
    for item in sorted(SOURCE.iterdir()):
        if item.is_dir() and (item / "test.csv").exists() and (item / "train.csv").exists():
            articles.append(item)
    return articles


def build_stochastic_sequence(n_periods=48):
    """One shared lead-time sequence for all stochastic instances."""
    random.seed(SEED + 999)
    return [random.choice(STOCHASTIC_VALUES) for _ in range(n_periods)]


def build_ph_assignments(n_instances=600):
    """Random (profit, holding_cost) per instance, seed=42."""
    random.seed(SEED)
    return [random.choice(PH_RATIOS) for _ in range(n_instances)]


def process_test_csv(
    src_test: Path,
    dest_test: Path,
    article_id: str,
    lead_time_value,  # 0, 4, or list of 48 values for stochastic
    profit: int,
    holding: int,
):
    """Read test.csv, set lead_time and p:h, write with proper quoting."""
    with open(src_test, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return
    header = rows[0]
    data_rows = rows[1:]

    # Header must be: exact_dates_{id}, demand_{id}, description_{id}, lead_time_{id}, profit_{id}, holding_cost_{id}
    # Keep same column names with article_id
    if len(header) < 6:
        raise ValueError(f"Expected 6 columns, got {len(header)} in {src_test}")

    use_stochastic = isinstance(lead_time_value, list)

    with open(dest_test, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        for i, row in enumerate(data_rows):
            if len(row) < 6:
                continue
            date_val = row[0]
            demand_val = row[1]
            desc_val = row[2]
            if use_stochastic:
                lt_val = lead_time_value[i] if i < len(lead_time_value) else lead_time_value[-1]
            else:
                lt_val = lead_time_value
            writer.writerow([date_val, demand_val, desc_val, lt_val, profit, holding])


def main():
    articles = get_article_dirs()
    if len(articles) != 200:
        print(f"Warning: expected 200 article dirs, found {len(articles)}")

    stochastic_sequence = build_stochastic_sequence()
    ph_list = build_ph_assignments(600)

    OUT.mkdir(parents=True, exist_ok=True)

    for lead_idx, lead_name in enumerate(LEAD_FOLDERS):
        lead_path = OUT / lead_name
        lead_path.mkdir(parents=True, exist_ok=True)
        if lead_name == "lead_time_stochastic":
            lead_val = stochastic_sequence
        elif lead_name == "lead_time_0":
            lead_val = 0
        else:
            lead_val = 4

        for art_idx, art_dir in enumerate(articles):
            article_id = art_dir.name
            instance_index = lead_idx * 200 + art_idx
            profit, holding = ph_list[instance_index]

            dest_dir = lead_path / article_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy train.csv unchanged
            shutil.copy2(art_dir / "train.csv", dest_dir / "train.csv")

            # Write test.csv with correct lead_time and p:h
            process_test_csv(
                art_dir / "test.csv",
                dest_dir / "test.csv",
                article_id,
                lead_val,
                profit,
                holding,
            )

        print(f"  {lead_name}: {len(articles)} instances")

    print("Done. real_trajectory has 600 instances (200 x 3 lead time settings).")


if __name__ == "__main__":
    main()
