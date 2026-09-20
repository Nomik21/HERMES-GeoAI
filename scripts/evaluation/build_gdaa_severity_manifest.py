from pathlib import Path
import csv
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"
OUTPUT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_full_manifest.csv"
OUTPUT_PILOT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_manifest.csv"

SEVERITY_SCENARIOS = {
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
}

# Small balanced pilot before running full 595.
# 2 cases per severity class per scenario.
PILOT_PER_GROUP = 2

FIELDNAMES = [
    "case_id",
    "dataset",
    "scenario",
    "image_paths",
    "source_types",
    "gt_hazard_type",
    "gt_severity_label",
    "gt_severity_id",
    "gt_pred_mode",
]


def has_severity(row):
    return str(row.get("gt_severity_label", "")).strip() != ""


def validate_paths(rows):
    missing = []

    for row in rows:
        for rel_path in str(row["image_paths"]).split(";"):
            rel_path = rel_path.strip()
            if not rel_path:
                continue

            abs_path = PROJECT_ROOT / rel_path
            if not abs_path.exists():
                missing.append(str(abs_path))

    return missing


with INPUT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

severity_rows = [
    row for row in rows
    if row["scenario"] in SEVERITY_SCENARIOS and has_severity(row)
]

with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()

    for row in severity_rows:
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})

# Build balanced pilot: scenario + gt_severity_label.
group_counts = Counter()
pilot_rows = []

for row in severity_rows:
    key = (row["scenario"], row["gt_severity_label"])

    if group_counts[key] >= PILOT_PER_GROUP:
        continue

    pilot_rows.append(row)
    group_counts[key] += 1

with OUTPUT_PILOT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()

    for row in pilot_rows:
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})

missing = validate_paths(severity_rows)

print(f"Input manifest: {INPUT_MANIFEST}")
print(f"Full GDAA severity manifest written to: {OUTPUT_MANIFEST}")
print(f"Pilot GDAA severity manifest written to: {OUTPUT_PILOT_MANIFEST}")
print()
print(f"Full rows written: {len(severity_rows)}")
print(f"Pilot rows written: {len(pilot_rows)}")

print("\nFull rows by scenario:")
scenario_counts = Counter(row["scenario"] for row in severity_rows)
for scenario, count in sorted(scenario_counts.items()):
    print(f"  {scenario}: {count}")

print("\nFull rows by scenario + severity:")
scenario_severity_counts = Counter(
    (row["scenario"], row["gt_severity_label"])
    for row in severity_rows
)
for (scenario, severity), count in sorted(scenario_severity_counts.items()):
    print(f"  {scenario} | {severity}: {count}")

print("\nPilot rows by scenario + severity:")
pilot_counts = Counter(
    (row["scenario"], row["gt_severity_label"])
    for row in pilot_rows
)
for (scenario, severity), count in sorted(pilot_counts.items()):
    print(f"  {scenario} | {severity}: {count}")

print("\nPath validation:")
print(f"  Missing paths: {len(missing)}")

if missing:
    for item in missing[:50]:
        print(f"  MISSING: {item}")
    raise SystemExit("Path validation failed ❌")

print("\nGDAA SEVERITY MANIFEST READY ✅")
