from pathlib import Path
import csv
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"
OUTPUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest_smoke.csv"

N_PER_GROUP = 2

with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

selected = []
groups = defaultdict(list)

for row in rows:
    scenario = row["scenario"]

    if scenario == "single_svi_hazard_classification":
        group_key = (scenario, row["gt_hazard_type"])
    else:
        group_key = (scenario, row["gt_severity_label"])

    groups[group_key].append(row)

for group_key, group_rows in sorted(groups.items()):
    selected.extend(group_rows[:N_PER_GROUP])

with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(selected)

print(f"Smoke manifest written to: {OUTPUT_CSV}")
print(f"Selected rows: {len(selected)}")

print("\nGroups:")
for group_key, group_rows in sorted(groups.items()):
    picked = min(N_PER_GROUP, len(group_rows))
    print(f"  {group_key}: {picked}/{len(group_rows)}")
