"""One-off: aggregate real_small_batch lead_time_0 benchmark results to CSV/table."""
import json
from pathlib import Path

base = Path(__file__).parent.parent / "examples/benchmark/real_small_batch/lead_time_0"
rows = []
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    j = d / "benchmark_results.json"
    if not j.exists():
        continue
    with open(j) as f:
        data = json.load(f)
    s = data.get("summary", {})
    ratios = s.get("ratios", {})
    ps = s.get("perfect_score") or 0

    def pct(name):
        v = ratios.get(name)
        return f"{v*100:.2f}%" if v is not None else "N/A"

    rows.append({
        "id": d.name,
        "perfect": ps,
        "or": pct("or"),
        "llm": pct("llm"),
        "llm_to_or": pct("llm_to_or"),
        "or_to_llm": pct("or_to_llm"),
    })

# CSV
print("ArticleID,PerfectScore,OR,LLM,LLM-to-OR,OR-to-LLM")
for row in rows:
    print(f"{row['id']},{row['perfect']:.0f},{row['or']},{row['llm']},{row['llm_to_or']},{row['or_to_llm']}")

# Averages (numeric)
def num_ratio(ratios, name):
    v = ratios.get(name)
    return v if v is not None else float("nan")

all_or = [num_ratio(data.get("summary", {}).get("ratios", {}), "or") for r in rows for (data,) in [(json.load(open(base / r["id"] / "benchmark_results.json")),)]]
# Re-read for averages
or_vals = []
llm_vals = []
l2o_vals = []
o2l_vals = []
for row in rows:
    with open(base / row["id"] / "benchmark_results.json") as f:
        r = json.load(f).get("summary", {}).get("ratios", {})
    if r.get("or") is not None:
        or_vals.append(r["or"])
    if r.get("llm") is not None:
        llm_vals.append(r["llm"])
    if r.get("llm_to_or") is not None:
        l2o_vals.append(r["llm_to_or"])
    if r.get("or_to_llm") is not None:
        o2l_vals.append(r["or_to_llm"])

import statistics
print("\n--- Averages (ratio to perfect) ---")
print(f"OR:       {statistics.mean(or_vals)*100:.2f}%  (n={len(or_vals)})")
print(f"LLM:      {statistics.mean(llm_vals)*100:.2f}%  (n={len(llm_vals)})")
print(f"LLM-to-OR: {statistics.mean(l2o_vals)*100:.2f}%  (n={len(l2o_vals)})")
print(f"OR-to-LLM: {statistics.mean(o2l_vals)*100:.2f}%  (n={len(o2l_vals)})")
