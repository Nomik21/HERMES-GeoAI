from pathlib import Path
import csv
import shutil
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"
OBJECT_GT_CSV = PROJECT_ROOT / "evaluation" / "hermes_object_groundtruth.csv"

OUT_ROOT = PROJECT_ROOT / "evaluation" / "report_eval_inputs_visual"
SELECTION_CSV = PROJECT_ROOT / "evaluation" / "hermes_report_eval_selection.csv"

SCENARIO_DIRS = {
    "cross_view_svi_rsi": "cross_view_svi_rsi",
    "pre_post_svi": "pre_post_svi",
    "wildfire_building": "wildfire_building",
    "object_detection": "object_detection",
}


# -----------------------------
# Helpers
# -----------------------------
def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_image_paths(value: str):
    if not value:
        return []
    return [p.strip() for p in value.split(";") if p.strip()]


def abs_path_from_rel(rel_path: str) -> Path:
    return PROJECT_ROOT / Path(rel_path)


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def positive_count(row):
    cols = ["Obj_Debris", "Obj_FallenTree", "Obj_Flooded", "Obj_DamagedBldg", "Obj_DownedLines"]
    return sum(safe_int(row.get(c, 0)) for c in cols)


def case_key(row):
    return (row["scenario"], row["case_id"])


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


# -----------------------------
# Selection logic
# -----------------------------
def select_severity_cases(manifest_rows, scenario, label_quota_pairs):
    scenario_rows = [r for r in manifest_rows if r["scenario"] == scenario]
    selected = []
    used = set()

    for label, quota in label_quota_pairs:
        candidates = [
            r for r in scenario_rows
            if r.get("gt_severity_label", "").strip() == label and r["case_id"] not in used
        ]
        candidates = sorted(candidates, key=lambda r: r["case_id"])
        picks = candidates[:quota]
        selected.extend(picks)
        used.update(r["case_id"] for r in picks)

    return selected


def select_object_detection_cases(object_rows):
    """
    Select 10 diverse object-detection cases.
    Priority:
      1. rare / informative indicators first
      2. unique cases only
      3. fill any remaining slots with cases having many positive indicators
    """
    selected = []
    used = set()

    target_plan = [
        ("Obj_DownedLines", 2),
        ("Obj_Flooded", 2),
        ("Obj_DamagedBldg", 2),
        ("Obj_FallenTree", 2),
        ("Obj_Debris", 2),
    ]

    for obj_col, quota in target_plan:
        candidates = [
            r for r in object_rows
            if safe_int(r.get(obj_col, 0)) == 1 and case_key(r) not in used
        ]
        candidates = sorted(
            candidates,
            key=lambda r: (-positive_count(r), r["scenario"], r["case_id"])
        )

        for r in candidates[:quota]:
            selected.append(r)
            used.add(case_key(r))

    if len(selected) < 10:
        remaining = [
            r for r in object_rows
            if case_key(r) not in used and positive_count(r) > 0
        ]
        remaining = sorted(
            remaining,
            key=lambda r: (-positive_count(r), r["scenario"], r["case_id"])
        )
        for r in remaining:
            if len(selected) >= 10:
                break
            selected.append(r)
            used.add(case_key(r))

    return selected[:10]


# -----------------------------
# Copy helpers
# -----------------------------
def copy_manifest_case(row, scenario_root: Path, rank: int, selection_rows: list, report_scenario: str):
    case_id = row["case_id"]
    case_folder = scenario_root / f"{rank:02d}__{case_id}"
    case_folder.mkdir(parents=True, exist_ok=True)

    rel_paths = split_image_paths(row["image_paths"])
    copied_names = []

    for rel in rel_paths:
        src = abs_path_from_rel(rel)
        dst = case_folder / Path(rel).name
        shutil.copy2(src, dst)
        copied_names.append(dst.name)

    info = []
    info.append(f"report_scenario: {report_scenario}")
    info.append(f"source_scenario: {row.get('scenario', '')}")
    info.append(f"case_id: {case_id}")
    info.append(f"dataset: {row.get('dataset', '')}")
    info.append(f"gt_hazard_type: {row.get('gt_hazard_type', '')}")
    info.append(f"gt_severity_label: {row.get('gt_severity_label', '')}")
    info.append(f"gt_severity_id: {row.get('gt_severity_id', '')}")
    info.append(f"gt_pred_mode: {row.get('gt_pred_mode', '')}")
    info.append(f"source_types: {row.get('source_types', '')}")
    info.append("")
    info.append("original_image_paths:")
    for rel in rel_paths:
        info.append(f"  - {rel}")

    write_text(case_folder / "case_info.txt", "\n".join(info))

    selection_rows.append({
        "report_scenario": report_scenario,
        "source_scenario": row.get("scenario", ""),
        "case_id": case_id,
        "dataset": row.get("dataset", ""),
        "gt_hazard_type": row.get("gt_hazard_type", ""),
        "gt_severity_label": row.get("gt_severity_label", ""),
        "gt_severity_id": row.get("gt_severity_id", ""),
        "gt_pred_mode": row.get("gt_pred_mode", ""),
        "source_types": row.get("source_types", ""),
        "copied_files": ";".join(copied_names),
        "folder": str(case_folder.relative_to(PROJECT_ROOT)),
    })


def copy_object_detection_case(obj_row, manifest_lookup, scenario_root: Path, rank: int, selection_rows: list):
    key = (obj_row["scenario"], obj_row["case_id"])
    man_row = manifest_lookup[key]

    case_id = obj_row["case_id"]
    case_folder = scenario_root / f"{rank:02d}__{case_id}"
    case_folder.mkdir(parents=True, exist_ok=True)

    rel_paths = split_image_paths(man_row["image_paths"])

    # For object_detection we use one visual input:
    # - cross_view_svi_rsi  -> SVI image (2nd)
    # - pre_post_svi        -> post image (2nd)
    # Fallback to first if needed.
    if len(rel_paths) >= 2:
        chosen_rel = rel_paths[1]
    elif len(rel_paths) == 1:
        chosen_rel = rel_paths[0]
    else:
        raise ValueError(f"No image path found for object-detection case: {case_id}")

    src = abs_path_from_rel(chosen_rel)
    dst = case_folder / Path(chosen_rel).name
    shutil.copy2(src, dst)

    obj_cols = ["Obj_Debris", "Obj_FallenTree", "Obj_Flooded", "Obj_DamagedBldg", "Obj_DownedLines"]
    info = []
    info.append("report_scenario: object_detection")
    info.append(f"source_scenario: {obj_row.get('scenario', '')}")
    info.append(f"case_id: {case_id}")
    info.append(f"dataset: {obj_row.get('dataset', '')}")
    info.append(f"gt_hazard_type: {man_row.get('gt_hazard_type', '')}")
    info.append(f"gt_pred_mode: {man_row.get('gt_pred_mode', '')}")
    info.append("")
    info.append("object_groundtruth:")
    for col in obj_cols:
        info.append(f"  {col}: {obj_row.get(col, '0')}")
    info.append("")
    info.append("chosen_input_image:")
    info.append(f"  - {chosen_rel}")
    info.append("")
    info.append("full_case_image_paths:")
    for rel in rel_paths:
        info.append(f"  - {rel}")

    write_text(case_folder / "case_info.txt", "\n".join(info))

    selection_rows.append({
        "report_scenario": "object_detection",
        "source_scenario": obj_row.get("scenario", ""),
        "case_id": case_id,
        "dataset": obj_row.get("dataset", ""),
        "gt_hazard_type": man_row.get("gt_hazard_type", ""),
        "gt_severity_label": "",
        "gt_severity_id": "",
        "gt_pred_mode": man_row.get("gt_pred_mode", ""),
        "source_types": "SV",
        "copied_files": dst.name,
        "folder": str(case_folder.relative_to(PROJECT_ROOT)),
    })


# -----------------------------
# Main
# -----------------------------
def main():
    if not DATASET_MANIFEST.exists():
        raise FileNotFoundError(f"Missing: {DATASET_MANIFEST}")
    if not OBJECT_GT_CSV.exists():
        raise FileNotFoundError(f"Missing: {OBJECT_GT_CSV}")

    manifest_rows = read_csv(DATASET_MANIFEST)
    object_rows = read_csv(OBJECT_GT_CSV)

    manifest_lookup = {(r["scenario"], r["case_id"]): r for r in manifest_rows}

    ensure_clean_dir(OUT_ROOT)

    # Create 4 scenario roots
    scenario_roots = {}
    for scenario, folder_name in SCENARIO_DIRS.items():
        p = OUT_ROOT / folder_name
        p.mkdir(parents=True, exist_ok=True)
        scenario_roots[scenario] = p

    selection_rows = []

    # 1) cross_view_svi_rsi -> 10 cases
    cross_selected = select_severity_cases(
        manifest_rows,
        "cross_view_svi_rsi",
        [("Mild", 4), ("Moderate", 3), ("Severe", 3)]
    )
    for i, row in enumerate(cross_selected, start=1):
        copy_manifest_case(row, scenario_roots["cross_view_svi_rsi"], i, selection_rows, "cross_view_svi_rsi")

    # 2) pre_post_svi -> 10 cases
    prepost_selected = select_severity_cases(
        manifest_rows,
        "pre_post_svi",
        [("Mild", 4), ("Moderate", 3), ("Severe", 3)]
    )
    for i, row in enumerate(prepost_selected, start=1):
        copy_manifest_case(row, scenario_roots["pre_post_svi"], i, selection_rows, "pre_post_svi")

    # 3) wildfire_building -> 10 cases (2 per class)
    wildfire_selected = select_severity_cases(
        manifest_rows,
        "wildfire_building",
        [
            ("0_No_Damage", 2),
            ("1_Affected_1_9", 2),
            ("2_Minor_10_25", 2),
            ("3_Major_26_50", 2),
            ("4_Destroyed_50plus", 2),
        ]
    )
    for i, row in enumerate(wildfire_selected, start=1):
        copy_manifest_case(row, scenario_roots["wildfire_building"], i, selection_rows, "wildfire_building")

    # 4) object_detection -> 10 cases from object GT
    object_selected = select_object_detection_cases(object_rows)
    for i, row in enumerate(object_selected, start=1):
        copy_object_detection_case(row, manifest_lookup, scenario_roots["object_detection"], i, selection_rows)

    # Write top-level selection CSV
    fieldnames = [
        "report_scenario",
        "source_scenario",
        "case_id",
        "dataset",
        "gt_hazard_type",
        "gt_severity_label",
        "gt_severity_id",
        "gt_pred_mode",
        "source_types",
        "copied_files",
        "folder",
    ]
    with SELECTION_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_rows)

    # Write README
    summary = []
    summary.append("HERMES Report Evaluation Visual Input Set")
    summary.append("=" * 60)
    summary.append("")
    summary.append(f"Output root: {OUT_ROOT}")
    summary.append(f"Selection CSV: {SELECTION_CSV}")
    summary.append("")
    summary.append("Scenarios:")
    summary.append("  - cross_view_svi_rsi: 10 cases")
    summary.append("  - pre_post_svi: 10 cases")
    summary.append("  - wildfire_building: 10 cases")
    summary.append("  - object_detection: 10 cases")
    summary.append("")
    summary.append("Total selected cases: 40")
    summary.append("")
    summary.append("Notes:")
    summary.append("  * cross_view_svi_rsi / pre_post_svi / wildfire_building were selected deterministically.")
    summary.append("  * object_detection was selected from hermes_object_groundtruth.csv with emphasis on indicator diversity.")
    summary.append("  * each case folder contains the copied input image(s) and a case_info.txt file.")

    write_text(OUT_ROOT / "README.txt", "\n".join(summary))

    # Print summary
    counts = defaultdict(int)
    for row in selection_rows:
        counts[row["report_scenario"]] += 1

    print(f"Visual input folder written to: {OUT_ROOT}")
    print(f"Selection CSV written to: {SELECTION_CSV}")
    print("Counts by scenario:")
    for k in ["cross_view_svi_rsi", "pre_post_svi", "wildfire_building", "object_detection"]:
        print(f"  {k}: {counts[k]}")
    print(f"Total cases: {sum(counts.values())}")
    print("DONE ✅")


if __name__ == "__main__":
    main()