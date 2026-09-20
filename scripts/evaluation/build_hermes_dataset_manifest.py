from pathlib import Path
import csv
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "hermes_benchmark"
OUTPUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

FIELDNAMES = [
    "case_id",
    "dataset",
    "scenario",
    "case_type",
    "image_paths",
    "image_path_1",
    "image_path_2",
    "source_types",
    "gt_hazard_type",
    "gt_hazard_raw",
    "gt_pred_mode",
    "gt_severity_label",
    "gt_severity_id",
    "gt_damage_class",
    "evaluation_task",
    "case_dir",
]

rows = []


def rel(path: Path) -> str:
    """Return project-relative path for reproducibility."""
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def image_files(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def add_row(**kwargs):
    row = {field: "" for field in FIELDNAMES}
    row.update(kwargs)
    rows.append(row)


def require_dir(path: Path):
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Required dataset folder not found: {path}")


# ---------------------------------------------------------------------
# 1. Cross-view SVI + satellite cases
# ---------------------------------------------------------------------
cross_view_root = DATASET_ROOT / "SVI & RSI_CVIAN_selected_50"
require_dir(cross_view_root)

cross_view_classes = {
    "0_MinorDamage": ("Mild", 0),
    "1_ModerateDamage": ("Moderate", 1),
    "2_SevereDamage": ("Severe", 2),
}

for class_name, (severity_label, severity_id) in cross_view_classes.items():
    class_dir = cross_view_root / class_name
    require_dir(class_dir)

    for case_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
        satellite = sorted(case_dir.glob("*_Satellite.*"))
        svi = sorted(case_dir.glob("*_SVI.*"))

        if len(satellite) != 1 or len(svi) != 1:
            raise RuntimeError(
                f"Invalid cross-view case: {case_dir} | "
                f"Satellite={len(satellite)}, SVI={len(svi)}"
            )

        image_path_1 = rel(satellite[0])
        image_path_2 = rel(svi[0])

        add_row(
            case_id=case_dir.name,
            dataset="CVIAN_selected_50",
            scenario="cross_view_svi_rsi",
            case_type="paired_post_event_rs_sv",
            image_paths=f"{image_path_1};{image_path_2}",
            image_path_1=image_path_1,
            image_path_2=image_path_2,
            source_types="RS;SV",
            gt_hazard_type="hurricane",
            gt_hazard_raw="hurricane",
            gt_pred_mode="post_paired_sv_rs",
            gt_severity_label=severity_label,
            gt_severity_id=severity_id,
            gt_damage_class="",
            evaluation_task="hazard+routing+severity+object_indicators+report",
            case_dir=rel(case_dir),
        )


# ---------------------------------------------------------------------
# 2. Pre/post street-view cases
# ---------------------------------------------------------------------
prepost_root = DATASET_ROOT / "pre and post SVI(50-50-50)"
require_dir(prepost_root)

prepost_classes = {
    "mild": ("Mild", 0),
    "moderate": ("Moderate", 1),
    "severe": ("Severe", 2),
}

for class_name, (severity_label, severity_id) in prepost_classes.items():
    class_dir = prepost_root / class_name
    require_dir(class_dir)

    for case_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
        pre = sorted(case_dir.glob("*_2023.*"))
        post = sorted(case_dir.glob("*_2024.*"))

        if len(pre) != 1 or len(post) != 1:
            raise RuntimeError(
                f"Invalid pre/post case: {case_dir} | "
                f"Pre2023={len(pre)}, Post2024={len(post)}"
            )

        image_path_1 = rel(pre[0])
        image_path_2 = rel(post[0])

        add_row(
            case_id=case_dir.name,
            dataset="pre_post_SVI_50_50_50",
            scenario="pre_post_svi",
            case_type="paired_bitemporal_sv",
            image_paths=f"{image_path_1};{image_path_2}",
            image_path_1=image_path_1,
            image_path_2=image_path_2,
            source_types="SV;SV",
            gt_hazard_type="hurricane",
            gt_hazard_raw="hurricane",
            gt_pred_mode="bitemporal_sv",
            gt_severity_label=severity_label,
            gt_severity_id=severity_id,
            gt_damage_class="",
            evaluation_task="hazard+routing+severity+object_indicators+report",
            case_dir=rel(case_dir),
        )


# ---------------------------------------------------------------------
# 3. Wildfire building damage cases
# ---------------------------------------------------------------------
wildfire_root = DATASET_ROOT / "SVI_PalisadesFireImages"
require_dir(wildfire_root)

wildfire_classes = {
    "0_No_Damage": ("0_No_Damage", 0),
    "1_Affected_1_9": ("1_Affected_1_9", 1),
    "2_Minor_10_25": ("2_Minor_10_25", 2),
    "3_Major_26_50": ("3_Major_26_50", 3),
    "4_Destroyed_50plus": ("4_Destroyed_50plus", 4),
}

for class_name, (damage_class, class_id) in wildfire_classes.items():
    class_dir = wildfire_root / class_name
    require_dir(class_dir)

    for img in image_files(class_dir):
        image_path_1 = rel(img)

        add_row(
            case_id=img.stem,
            dataset="SVI_PalisadesFireImages",
            scenario="wildfire_building",
            case_type="single_post_event_sv",
            image_paths=image_path_1,
            image_path_1=image_path_1,
            image_path_2="",
            source_types="SV",
            gt_hazard_type="wildfire",
            gt_hazard_raw="wildfire",
            gt_pred_mode="post_sv_single",
            gt_severity_label=damage_class,
            gt_severity_id=class_id,
            gt_damage_class=damage_class,
            evaluation_task="hazard+routing+wildfire_severity+report",
            case_dir=rel(class_dir),
        )


# ---------------------------------------------------------------------
# 4. Single-image multi-hazard cases
# ---------------------------------------------------------------------
incidents_root = DATASET_ROOT / "SVI_IncidentsDataset"
require_dir(incidents_root)

hazard_map = {
    "drought": "drought",
    "earthquake": "earthquake",
    "flooded": "flood",
    "ice_storm": "ice_storm",
    "tropical_cyclone": "hurricane",
    "wildfire": "wildfire",
}

for raw_hazard, normalized_hazard in hazard_map.items():
    class_dir = incidents_root / raw_hazard
    require_dir(class_dir)

    for img in image_files(class_dir):
        image_path_1 = rel(img)

        add_row(
            case_id=f"{raw_hazard}_{img.stem}",
            dataset="SVI_IncidentsDataset",
            scenario="single_svi_hazard_classification",
            case_type="single_post_event_sv",
            image_paths=image_path_1,
            image_path_1=image_path_1,
            image_path_2="",
            source_types="SV",
            gt_hazard_type=normalized_hazard,
            gt_hazard_raw=raw_hazard,
            gt_pred_mode="post_sv_single",
            gt_severity_label="",
            gt_severity_id="",
            gt_damage_class="",
            evaluation_task="hazard+routing",
            case_dir=rel(class_dir),
        )


# ---------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Manifest written to: {OUTPUT_CSV}")
print(f"Total rows: {len(rows)}")

print("\nRows by dataset:")
for key, count in Counter(r["dataset"] for r in rows).items():
    print(f"  {key}: {count}")

print("\nRows by scenario:")
for key, count in Counter(r["scenario"] for r in rows).items():
    print(f"  {key}: {count}")

print("\nRows by gt_hazard_type:")
for key, count in Counter(r["gt_hazard_type"] for r in rows).items():
    print(f"  {key}: {count}")

print("\nRows by gt_severity_label:")
for key, count in Counter(r["gt_severity_label"] or "N/A" for r in rows).items():
    print(f"  {key}: {count}")
