from pathlib import Path
import csv
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest_smoke.csv"
NEUTRAL_DIR = PROJECT_ROOT / "evaluation" / "gpa_neutral_inputs" / "smoke"
OUTPUT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_gpa_hazard_smoke_neutral_manifest.csv"

TARGET_SCENARIO = "single_svi_hazard_classification"

HAZARD_TERMS = [
    "drought",
    "earthquake",
    "flood",
    "flooded",
    "hurricane",
    "ice_storm",
    "ice-storm",
    "tropical_cyclone",
    "cyclone",
    "wildfire",
]

FIELDNAMES = [
    "case_id",
    "original_case_id",
    "dataset",
    "scenario",
    "image_paths",
    "original_image_paths",
    "source_types",
    "gt_hazard_type",
    "gt_severity_label",
    "gt_severity_id",
    "gt_pred_mode",
]


def normalize_hazard(value):
    text = str(value or "").strip().lower()

    aliases = {
        "flooded": "flood",
        "flooding": "flood",
        "flood": "flood",
        "tropical_cyclone": "hurricane",
        "cyclone": "hurricane",
        "hurricane": "hurricane",
        "wildfire": "wildfire",
        "fire": "wildfire",
        "earthquake": "earthquake",
        "drought": "drought",
        "ice storm": "ice_storm",
        "ice_storm": "ice_storm",
        "ice-storm": "ice_storm",
    }

    return aliases.get(text, text)


def has_hazard_leakage(path_text):
    lowered = str(path_text).lower()
    return any(term in lowered for term in HAZARD_TERMS)


with INPUT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    all_rows = list(csv.DictReader(f))

target_rows = [
    row for row in all_rows
    if row["scenario"] == TARGET_SCENARIO
]

NEUTRAL_DIR.mkdir(parents=True, exist_ok=True)

# Clean only previous neutral smoke files created by this script.
for old_file in NEUTRAL_DIR.glob("gpa_smoke_*"):
    if old_file.is_file():
        old_file.unlink()

output_rows = []

for idx, row in enumerate(target_rows, start=1):
    original_rel_path = row["image_paths"]
    original_abs_path = PROJECT_ROOT / original_rel_path

    if not original_abs_path.exists():
        raise FileNotFoundError(f"Missing source image: {original_abs_path}")

    suffix = original_abs_path.suffix.lower()
    if not suffix:
        suffix = ".png"

    neutral_case_id = f"gpa_smoke_{idx:04d}"
    neutral_filename = f"{neutral_case_id}{suffix}"
    neutral_abs_path = NEUTRAL_DIR / neutral_filename

    shutil.copy2(original_abs_path, neutral_abs_path)

    neutral_rel_path = neutral_abs_path.relative_to(PROJECT_ROOT)

    out = {
        "case_id": neutral_case_id,
        "original_case_id": row["case_id"],
        "dataset": row["dataset"],
        "scenario": row["scenario"],
        "image_paths": str(neutral_rel_path),
        "original_image_paths": row["image_paths"],
        "source_types": row["source_types"],
        "gt_hazard_type": normalize_hazard(row["gt_hazard_type"]),
        "gt_severity_label": row["gt_severity_label"],
        "gt_severity_id": row["gt_severity_id"],
        "gt_pred_mode": row["gt_pred_mode"],
    }

    output_rows.append(out)

with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Neutral GPA smoke manifest written to: {OUTPUT_MANIFEST}")
print(f"Neutral image directory: {NEUTRAL_DIR}")
print(f"Rows written: {len(output_rows)}")

print("\nLeakage check on neutral image_paths:")
leaky = []

for row in output_rows:
    if has_hazard_leakage(row["image_paths"]):
        leaky.append(row["image_paths"])

print(f"  paths with hazard terms: {len(leaky)}")

if leaky:
    for item in leaky:
        print(f"  LEAK: {item}")
    raise SystemExit("Neutral path leakage detected ❌")

print("\nRows by gt_hazard_type:")
counts = {}
for row in output_rows:
    counts[row["gt_hazard_type"]] = counts.get(row["gt_hazard_type"], 0) + 1

for hazard, count in sorted(counts.items()):
    print(f"  {hazard}: {count}")

print("\nFirst rows:")
for row in output_rows[:6]:
    print(
        f"  {row['case_id']} | gt={row['gt_hazard_type']} | "
        f"neutral_path={row['image_paths']}"
    )

print("\nNEUTRAL GPA SMOKE MANIFEST READY ✅")
