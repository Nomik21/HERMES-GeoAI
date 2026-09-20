from pathlib import Path
import csv
import shutil
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"
NEUTRAL_DIR = PROJECT_ROOT / "evaluation" / "gpa_neutral_inputs" / "mode_full"
OUTPUT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_manifest.csv"

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
    "num_images",
]

LEAKAGE_TERMS = [
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
    "tornado",
    "landslide",
    "other",
    "mild",
    "moderate",
    "severe",
    "minor",
    "major",
    "destroyed",
    "affected",
    "damage",
    "satellite",
    "svi",
    "rsi",
    "pre",
    "post",
    "bitemporal",
    "paired",
    "cross_view",
    "palisades",
    "incidents",
]


def split_paths(path_text):
    return [p.strip() for p in str(path_text or "").split(";") if p.strip()]


def has_leakage(path_text):
    lowered = str(path_text).lower()
    return any(term in lowered for term in LEAKAGE_TERMS)


with INPUT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

NEUTRAL_DIR.mkdir(parents=True, exist_ok=True)

for old_file in NEUTRAL_DIR.glob("gpa_mode_full_*"):
    if old_file.is_file():
        old_file.unlink()

output_rows = []

for idx, row in enumerate(rows, start=1):
    original_paths = split_paths(row["image_paths"])

    if not original_paths:
        raise ValueError(f"No image paths for row {idx}: {row.get('case_id')}")

    neutral_case_id = f"gpa_mode_full_{idx:04d}"
    neutral_paths = []

    for image_index, original_rel_path in enumerate(original_paths):
        original_abs_path = PROJECT_ROOT / original_rel_path

        if not original_abs_path.exists():
            raise FileNotFoundError(f"Missing source image: {original_abs_path}")

        suffix = original_abs_path.suffix.lower() or ".jpg"
        neutral_filename = f"{neutral_case_id}_i{image_index}{suffix}"
        neutral_abs_path = NEUTRAL_DIR / neutral_filename

        shutil.copy2(original_abs_path, neutral_abs_path)

        neutral_rel_path = neutral_abs_path.relative_to(PROJECT_ROOT)
        neutral_paths.append(str(neutral_rel_path))

    output_rows.append({
        "case_id": neutral_case_id,
        "original_case_id": row["case_id"],
        "dataset": row["dataset"],
        "scenario": row["scenario"],
        "image_paths": ";".join(neutral_paths),
        "original_image_paths": row["image_paths"],
        "source_types": row["source_types"],
        "gt_hazard_type": row["gt_hazard_type"],
        "gt_severity_label": row["gt_severity_label"],
        "gt_severity_id": row["gt_severity_id"],
        "gt_pred_mode": row["gt_pred_mode"],
        "num_images": len(neutral_paths),
    })

with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Full neutral GPA mode manifest written to: {OUTPUT_MANIFEST}")
print(f"Neutral image directory: {NEUTRAL_DIR}")
print(f"Rows written: {len(output_rows)}")

print("\nLeakage check on neutral image_paths:")
leaky = [row["image_paths"] for row in output_rows if has_leakage(row["image_paths"])]
print(f"  paths with leakage terms: {len(leaky)}")

if leaky:
    for item in leaky[:50]:
        print(f"  LEAK: {item}")
    raise SystemExit("Neutral path leakage detected ❌")

print("\nRows by gt_pred_mode:")
mode_counts = Counter(row["gt_pred_mode"] for row in output_rows)
for mode, count in sorted(mode_counts.items()):
    print(f"  {mode}: {count}")

print("\nRows by scenario:")
scenario_counts = Counter(row["scenario"] for row in output_rows)
for scenario, count in sorted(scenario_counts.items()):
    print(f"  {scenario}: {count}")

print("\nFirst rows:")
for row in output_rows[:8]:
    print(
        f"  {row['case_id']} | gt_mode={row['gt_pred_mode']} | "
        f"scenario={row['scenario']} | images={row['num_images']} | "
        f"path={row['image_paths']}"
    )

print("\nFULL NEUTRAL GPA MODE MANIFEST READY ✅")
