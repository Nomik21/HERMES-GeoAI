from pathlib import Path
import csv
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "dataset_manifest": PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv",
    "gdaa_severity_full_manifest": PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_full_manifest.csv",
    "gdaa_severity_pilot_manifest": PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_manifest.csv",
    "object_groundtruth": PROJECT_ROOT / "evaluation" / "hermes_object_groundtruth.csv",
}

OUT_TXT = PROJECT_ROOT / "evaluation" / "hermes_groundtruth_audit_summary.txt"
OUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_groundtruth_label_inventory.csv"

OBJECT_COLUMNS = [
    "Obj_Debris",
    "Obj_FallenTree",
    "Obj_Flooded",
    "Obj_DamagedBldg",
    "Obj_DownedLines",
]

CORE_GT_COLUMNS = [
    "scenario",
    "dataset",
    "gt_hazard_type",
    "gt_severity_label",
    "gt_severity_id",
    "gt_pred_mode",
    "source_types",
]


def read_csv(path):
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []


def clean(value):
    return str(value or "").strip()


def split_semicolon(value):
    return [x.strip() for x in str(value or "").split(";") if x.strip()]


def value_counts(rows, col):
    c = Counter()
    missing = 0

    for row in rows:
        value = clean(row.get(col, ""))

        if value == "":
            missing += 1
        else:
            c[value] += 1

    return c, missing


def print_counter(lines, title, counter, missing=None, indent="  "):
    lines.append(title)

    if not counter:
        lines.append(f"{indent}<none>")

    for key, count in sorted(counter.items(), key=lambda x: (str(x[0]), x[1])):
        lines.append(f"{indent}{key}: {count}")

    if missing is not None:
        lines.append(f"{indent}<missing/empty>: {missing}")

    lines.append("")


def add_inventory(inventory, source_file, column, label, count, scenario="", extra=""):
    inventory.append({
        "source_file": source_file,
        "scenario": scenario,
        "column": column,
        "label_or_value": label,
        "count": count,
        "extra": extra,
    })


def validate_image_paths(rows):
    missing_paths = []

    for row in rows:
        for rel_path in split_semicolon(row.get("image_paths", "")):
            path = Path(rel_path)

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            if not path.exists():
                missing_paths.append(str(path))

    return missing_paths


def audit_label_id_consistency(rows):
    issues = []

    by_scenario_label = defaultdict(set)
    by_scenario_id = defaultdict(set)

    for row in rows:
        scenario = clean(row.get("scenario"))
        label = clean(row.get("gt_severity_label"))
        sid = clean(row.get("gt_severity_id"))

        if not scenario or not label:
            continue

        by_scenario_label[(scenario, label)].add(sid)
        by_scenario_id[(scenario, sid)].add(label)

    for (scenario, label), ids in sorted(by_scenario_label.items()):
        if len(ids) > 1:
            issues.append(
                f"Label maps to multiple IDs: scenario={scenario} | label={label} | ids={sorted(ids)}"
            )

    for (scenario, sid), labels in sorted(by_scenario_id.items()):
        if sid != "" and len(labels) > 1:
            issues.append(
                f"ID maps to multiple labels: scenario={scenario} | id={sid} | labels={sorted(labels)}"
            )

    return issues, by_scenario_label, by_scenario_id


def audit_object_gt(rows, fieldnames):
    issues = []

    available_object_cols = [c for c in OBJECT_COLUMNS if c in fieldnames]

    for col in available_object_cols:
        bad_values = Counter()

        for row in rows:
            value = clean(row.get(col, ""))

            if value not in {"0", "1"}:
                bad_values[value] += 1

        if bad_values:
            issues.append(f"Non-binary values in {col}: {dict(bad_values)}")

    return available_object_cols, issues


def main():
    lines = []
    inventory = []

    lines.append("HERMES Ground-Truth Audit")
    lines.append("=" * 90)
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    loaded = {}

    for name, path in FILES.items():
        rows, fieldnames = read_csv(path)
        loaded[name] = (path, rows, fieldnames)

        lines.append("-" * 90)
        lines.append(f"{name}")
        lines.append("-" * 90)
        lines.append(f"Path: {path}")
        lines.append(f"Exists: {path.exists()}")
        lines.append(f"Rows: {len(rows)}")
        lines.append(f"Columns: {', '.join(fieldnames) if fieldnames else '<none>'}")
        lines.append("")

        if not rows:
            continue

        for col in CORE_GT_COLUMNS:
            if col not in fieldnames:
                continue

            counter, missing = value_counts(rows, col)
            print_counter(lines, f"{col} distribution:", counter, missing)

            for label, count in counter.items():
                add_inventory(inventory, name, col, label, count)

        # Per-scenario severity distributions
        if "scenario" in fieldnames and "gt_severity_label" in fieldnames:
            combo = Counter(
                (
                    clean(row.get("scenario")),
                    clean(row.get("gt_severity_label")),
                    clean(row.get("gt_severity_id")),
                )
                for row in rows
                if clean(row.get("gt_severity_label")) != ""
            )

            lines.append("scenario + gt_severity_label + gt_severity_id distribution:")
            if not combo:
                lines.append("  <none>")

            for (scenario, label, sid), count in sorted(combo.items()):
                lines.append(f"  {scenario} | {label} | id={sid}: {count}")
                add_inventory(
                    inventory,
                    name,
                    "gt_severity_label",
                    label,
                    count,
                    scenario=scenario,
                    extra=f"gt_severity_id={sid}",
                )

            lines.append("")

            issues, by_label, by_id = audit_label_id_consistency(rows)

            lines.append("severity label/id consistency:")
            if issues:
                for issue in issues:
                    lines.append(f"  ISSUE: {issue}")
            else:
                lines.append("  OK: no label/id consistency issues detected")
            lines.append("")

        # Per-scenario hazard distributions
        if "scenario" in fieldnames and "gt_hazard_type" in fieldnames:
            combo = Counter(
                (
                    clean(row.get("scenario")),
                    clean(row.get("gt_hazard_type")),
                )
                for row in rows
                if clean(row.get("gt_hazard_type")) != ""
            )

            lines.append("scenario + gt_hazard_type distribution:")
            if not combo:
                lines.append("  <none>")

            for (scenario, hazard), count in sorted(combo.items()):
                lines.append(f"  {scenario} | {hazard}: {count}")
                add_inventory(
                    inventory,
                    name,
                    "gt_hazard_type",
                    hazard,
                    count,
                    scenario=scenario,
                )

            lines.append("")

        # Per-scenario mode distributions
        if "scenario" in fieldnames and "gt_pred_mode" in fieldnames:
            combo = Counter(
                (
                    clean(row.get("scenario")),
                    clean(row.get("gt_pred_mode")),
                )
                for row in rows
                if clean(row.get("gt_pred_mode")) != ""
            )

            lines.append("scenario + gt_pred_mode distribution:")
            if not combo:
                lines.append("  <none>")

            for (scenario, mode), count in sorted(combo.items()):
                lines.append(f"  {scenario} | {mode}: {count}")
                add_inventory(
                    inventory,
                    name,
                    "gt_pred_mode",
                    mode,
                    count,
                    scenario=scenario,
                )

            lines.append("")

        # Object ground truth audit
        available_object_cols, object_issues = audit_object_gt(rows, fieldnames)

        if available_object_cols:
            lines.append("object ground-truth columns:")

            for col in available_object_cols:
                counter, missing = value_counts(rows, col)
                positives = counter.get("1", 0)
                negatives = counter.get("0", 0)

                lines.append(
                    f"  {col}: positives={positives} | negatives={negatives} | missing={missing}"
                )

                add_inventory(inventory, name, col, "positive_1", positives)
                add_inventory(inventory, name, col, "negative_0", negatives)
                add_inventory(inventory, name, col, "missing", missing)

            lines.append("")

            lines.append("object ground-truth binary consistency:")
            if object_issues:
                for issue in object_issues:
                    lines.append(f"  ISSUE: {issue}")
            else:
                lines.append("  OK: all object GT columns are binary 0/1")
            lines.append("")

        # Path validation
        if "image_paths" in fieldnames:
            missing_paths = validate_image_paths(rows)
            lines.append("image path validation:")
            lines.append(f"  missing paths: {len(missing_paths)}")

            for item in missing_paths[:30]:
                lines.append(f"  MISSING: {item}")

            lines.append("")

    # Cross-file consistency checks
    lines.append("=" * 90)
    lines.append("Cross-file checks")
    lines.append("=" * 90)

    dataset_rows = loaded["dataset_manifest"][1]
    severity_rows = loaded["gdaa_severity_full_manifest"][1]
    object_rows = loaded["object_groundtruth"][1]

    severity_case_ids = {clean(r.get("case_id")) for r in severity_rows}
    dataset_severity_case_ids = {
        clean(r.get("case_id"))
        for r in dataset_rows
        if clean(r.get("gt_severity_label")) != ""
    }

    lines.append("Severity manifest vs dataset manifest:")
    lines.append(f"  severity full manifest cases: {len(severity_case_ids)}")
    lines.append(f"  dataset manifest cases with severity GT: {len(dataset_severity_case_ids)}")
    lines.append(f"  missing from severity manifest: {len(dataset_severity_case_ids - severity_case_ids)}")
    lines.append(f"  extra in severity manifest: {len(severity_case_ids - dataset_severity_case_ids)}")
    lines.append("")

    if object_rows:
        object_case_ids = {clean(r.get("case_id")) for r in object_rows}
        dataset_case_ids = {clean(r.get("case_id")) for r in dataset_rows}

        lines.append("Object GT vs dataset manifest:")
        lines.append(f"  object GT cases: {len(object_case_ids)}")
        lines.append(f"  object GT cases not in dataset manifest: {len(object_case_ids - dataset_case_ids)}")
        lines.append(f"  dataset cases with object GT: {len(object_case_ids & dataset_case_ids)}")
        lines.append("")

    # Write outputs
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["source_file", "scenario", "column", "label_or_value", "count", "extra"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)

    print("\n".join(lines))
    print(f"\nAudit TXT written to: {OUT_TXT}")
    print(f"Inventory CSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
