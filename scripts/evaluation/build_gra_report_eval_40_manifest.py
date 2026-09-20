from pathlib import Path
import csv
import random
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"
OBJECT_GT = PROJECT_ROOT / "evaluation" / "hermes_object_groundtruth.csv"

OUT_ROOT = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set"
OUT_MANIFEST = OUT_ROOT / "gra_report_eval_40_manifest.csv"

RANDOM_SEED = 42

SCENARIOS = [
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
    "object_detection",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_by_severity(rows, scenario, plan):
    selected = []

    for label, count in plan:
        candidates = [
            r for r in rows
            if r.get("scenario") == scenario
            and r.get("gt_severity_label") == label
        ]
        candidates = sorted(candidates, key=lambda r: r["case_id"])
        selected.extend(candidates[:count])

    return selected


def safe_int(value):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def object_positive_count(row):
    cols = [
        "Obj_Debris",
        "Obj_FallenTree",
        "Obj_Flooded",
        "Obj_DamagedBldg",
        "Obj_DownedLines",
    ]
    return sum(safe_int(row.get(c, 0)) for c in cols)


def split_paths(value):
    return [p.strip() for p in str(value or "").split(";") if p.strip()]


def build_object_detection_rows(object_rows, manifest_lookup):
    selected = []
    used = set()

    target_plan = [
        ("Obj_DownedLines", 2),
        ("Obj_Flooded", 2),
        ("Obj_DamagedBldg", 2),
        ("Obj_FallenTree", 2),
        ("Obj_Debris", 2),
    ]

    for col, count in target_plan:
        candidates = [
            r for r in object_rows
            if safe_int(r.get(col, 0)) == 1
            and (r["scenario"], r["case_id"]) not in used
        ]

        candidates = sorted(
            candidates,
            key=lambda r: (-object_positive_count(r), r["scenario"], r["case_id"])
        )

        for r in candidates[:count]:
            selected.append(r)
            used.add((r["scenario"], r["case_id"]))

    if len(selected) < 10:
        remaining = [
            r for r in object_rows
            if (r["scenario"], r["case_id"]) not in used
        ]
        remaining = sorted(
            remaining,
            key=lambda r: (-object_positive_count(r), r["scenario"], r["case_id"])
        )

        for r in remaining:
            if len(selected) >= 10:
                break
            selected.append(r)
            used.add((r["scenario"], r["case_id"]))

    out_rows = []

    for idx, obj_row in enumerate(selected[:10], start=1):
        original_key = (obj_row["scenario"], obj_row["case_id"])
        man_row = manifest_lookup[original_key]

        paths = split_paths(man_row["image_paths"])

        # Object detection uses one image:
        # cross_view_svi_rsi -> SVI image, usually second
        # pre_post_svi -> post image, usually second
        if len(paths) >= 2:
            chosen_path = paths[1]
        elif len(paths) == 1:
            chosen_path = paths[0]
        else:
            raise ValueError(f"No image path for object_detection case: {obj_row['case_id']}")

        case_id = f"object_detection_{idx:02d}_{obj_row['case_id']}"

        out = dict(man_row)
        out["case_id"] = case_id
        out["scenario"] = "object_detection"
        out["case_type"] = "object_detection"
        out["image_paths"] = chosen_path
        out["image_path_1"] = chosen_path
        out["image_path_2"] = ""
        out["source_types"] = "SV"
        out["gt_pred_mode"] = "post_sv_single"
        out["gt_severity_label"] = ""
        out["gt_severity_id"] = ""
        out["gt_damage_class"] = ""
        out["evaluation_task"] = "gra_report_evaluation_object_detection"
        out["case_dir"] = ""

        out_rows.append(out)

    return out_rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Create visual/report folders
    for scenario in SCENARIOS:
        (OUT_ROOT / scenario / "inputs").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / scenario / "reports").mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(DATASET_MANIFEST)
    object_rows = read_csv(OBJECT_GT)

    manifest_lookup = {
        (r["scenario"], r["case_id"]): r
        for r in manifest_rows
    }

    selected = []

    selected.extend(
        choose_by_severity(
            manifest_rows,
            "cross_view_svi_rsi",
            [("Mild", 4), ("Moderate", 3), ("Severe", 3)],
        )
    )

    selected.extend(
        choose_by_severity(
            manifest_rows,
            "pre_post_svi",
            [("Mild", 4), ("Moderate", 3), ("Severe", 3)],
        )
    )

    selected.extend(
        choose_by_severity(
            manifest_rows,
            "wildfire_building",
            [
                ("0_No_Damage", 2),
                ("1_Affected_1_9", 2),
                ("2_Minor_10_25", 2),
                ("3_Major_26_50", 2),
                ("4_Destroyed_50plus", 2),
            ],
        )
    )

    selected.extend(
        build_object_detection_rows(object_rows, manifest_lookup)
    )

    fieldnames = [
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

    cleaned = []
    for row in selected:
        cleaned.append({k: row.get(k, "") for k in fieldnames})

    write_csv(OUT_MANIFEST, cleaned, fieldnames)

    print(f"Manifest written to: {OUT_MANIFEST}")
    print(f"Total cases: {len(cleaned)}")

    counts = defaultdict(int)
    for row in cleaned:
        counts[row["scenario"]] += 1

    print("Counts by scenario:")
    for scenario in SCENARIOS:
        print(f"  {scenario}: {counts[scenario]}")

    print("DONE ✅")


if __name__ == "__main__":
    main()
