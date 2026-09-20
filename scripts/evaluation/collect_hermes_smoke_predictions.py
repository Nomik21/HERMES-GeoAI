from pathlib import Path
import csv
import json
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_RESULTS_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_final_run_results.csv"
OUTPUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_final_predictions.csv"


FIELDNAMES = [
    "case_id",
    "dataset",
    "scenario",
    "gt_hazard_type",
    "pred_hazard_type",
    "hazard_correct",
    "gt_pred_mode",
    "pred_mode",
    "pred_mode_correct",
    "gt_severity_label",
    "gt_severity_id",
    "pred_severity_label",
    "pred_damage_class",
    "severity_match",
    "analysis_mode",
    "confidence",
    "damage_presence",
    "restoration_needed",
    "restoration_accepted",
    "mean_q_before",
    "mean_q_after",
    "mean_delta_q",
    "objects_detected_json",
    "final_report_guard_applied",
    "intermediate_state_used",
    "report_json_path",
]


def read_json(path: Path):
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def nested_get(data, path, default=None):
    current = data

    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def first_found(data, keys):
    if isinstance(data, dict):
        for key in keys:
            if key in data and data[key] not in [None, "", [], {}]:
                return data[key]

        for value in data.values():
            found = first_found(value, keys)
            if found not in [None, "", [], {}]:
                return found

    elif isinstance(data, list):
        for item in data:
            found = first_found(item, keys)
            if found not in [None, "", [], {}]:
                return found

    return None


def to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def bool_summary(value):
    values = to_list(value)

    if not values:
        return ""

    normalized = []
    for item in values:
        if isinstance(item, bool):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(item.strip().lower() in {"true", "1", "yes"})
        else:
            normalized.append(bool(item))

    if all(normalized):
        return "all_true"
    if not any(normalized):
        return "all_false"
    return "mixed"


def mean_numeric(value):
    values = to_list(value)
    nums = []

    for item in values:
        try:
            nums.append(float(item))
        except (TypeError, ValueError):
            pass

    if not nums:
        return ""

    return round(mean(nums), 6)


def normalize_severity(value):
    if value is None:
        return ""

    text = str(value).strip()

    mapping = {
        "minor": "Mild",
        "mild": "Mild",
        "moderate": "Moderate",
        "severe": "Severe",
    }

    return mapping.get(text.lower(), text)


def extract_case_part(image_paths):
    stems = []
    for rel_path in image_paths.split(";"):
        rel_path = rel_path.strip()
        if rel_path:
            stems.append(Path(rel_path).stem)
    return "__".join(stems)


def intermediate_path(case_part, agent_key):
    return PROJECT_ROOT / "outputs" / "intermediate" / f"{case_part}_{agent_key}.json"


def main():
    with RUN_RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        run_rows = list(csv.DictReader(f))

    success_rows = [
        row for row in run_rows
        if row.get("status") == "success" and row.get("report_exists") == "True"
    ]

    output_rows = []

    for row in success_rows:
        report_path = PROJECT_ROOT / row["report_json_path"]
        report = read_json(report_path)

        case_part = extract_case_part(row["image_paths"])

        gpa = read_json(intermediate_path(case_part, "gpa"))
        gira = read_json(intermediate_path(case_part, "gira"))
        gdaa = read_json(intermediate_path(case_part, "gdaa"))

        pred_hazard = (
            nested_get(gpa, ["mode_and_hazard", "hazard_type"])
            or first_found(report, ["hazard_type"])
            or first_found(gdaa, ["hazard_type"])
            or ""
        )

        pred_mode = (
            nested_get(gpa, ["mode_and_hazard", "pred_mode"])
            or first_found(report, ["pred_mode"])
            or ""
        )

        analysis_mode = (
            first_found(gdaa, ["analysis_mode"])
            or first_found(report, ["analysis_mode"])
            or ""
        )

        pred_severity = (
            first_found(gdaa, ["overall_severity", "severity", "three_level_severity"])
            or first_found(report, ["overall_severity", "severity", "three_level_severity"])
            or ""
        )

        pred_damage_class = (
            first_found(gdaa, ["damage_class", "wildfire_damage_class", "domain_damage_class"])
            or first_found(report, ["damage_class", "wildfire_damage_class", "domain_damage_class"])
            or ""
        )

        confidence = (
            first_found(gdaa, ["confidence"])
            or first_found(report, ["confidence"])
            or ""
        )

        damage_presence = (
            first_found(gdaa, ["damage_presence"])
            or first_found(report, ["damage_presence"])
            or ""
        )

        objects_detected = (
            first_found(gdaa, ["objects_detected", "object_counts"])
            or first_found(report, ["objects_detected", "object_counts"])
            or {}
        )

        restoration_needed = first_found(gira, ["needs_restoration"])
        restoration_accepted = first_found(gira, ["accepted"])
        q_before = first_found(gira, ["Q_before"])
        q_after = first_found(gira, ["Q_after"])
        delta_q = first_found(gira, ["delta_Q"])

        final_report_guard_applied = first_found(report, ["final_report_guard_applied"])
        intermediate_state_used = first_found(report, ["intermediate_state_used"])

        gt_severity = normalize_severity(row["gt_severity_label"])
        pred_severity_norm = normalize_severity(pred_severity)

        if row["scenario"] == "wildfire_building":
            severity_match = str(pred_damage_class == row["gt_severity_label"])
        else:
            severity_match = str(pred_severity_norm == gt_severity)

        output_rows.append(
            {
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                "gt_hazard_type": row["gt_hazard_type"],
                "pred_hazard_type": pred_hazard,
                "hazard_correct": str(pred_hazard == row["gt_hazard_type"]),
                "gt_pred_mode": row["gt_pred_mode"],
                "pred_mode": pred_mode,
                "pred_mode_correct": str(pred_mode == row["gt_pred_mode"]),
                "gt_severity_label": row["gt_severity_label"],
                "gt_severity_id": row["gt_severity_id"],
                "pred_severity_label": pred_severity_norm,
                "pred_damage_class": pred_damage_class,
                "severity_match": severity_match,
                "analysis_mode": analysis_mode,
                "confidence": confidence,
                "damage_presence": damage_presence,
                "restoration_needed": bool_summary(restoration_needed),
                "restoration_accepted": bool_summary(restoration_accepted),
                "mean_q_before": mean_numeric(q_before),
                "mean_q_after": mean_numeric(q_after),
                "mean_delta_q": mean_numeric(delta_q),
                "objects_detected_json": json.dumps(objects_detected, ensure_ascii=False),
                "final_report_guard_applied": final_report_guard_applied,
                "intermediate_state_used": intermediate_state_used,
                "report_json_path": row["report_json_path"],
            }
        )

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Predictions written to: {OUTPUT_CSV}")
    print(f"Rows written: {len(output_rows)}")

    print("\nRows by scenario:")
    counts = {}
    for r in output_rows:
        key = r["scenario"]
        counts[key] = counts.get(key, 0) + 1

    for key, count in counts.items():
        print(f"  {key}: {count}")

    print("\nQuick checks:")
    print(f"  hazard correct: {sum(r['hazard_correct'] == 'True' for r in output_rows)}/{len(output_rows)}")
    print(f"  pred_mode correct: {sum(r['pred_mode_correct'] == 'True' for r in output_rows)}/{len(output_rows)}")
    print(f"  severity match: {sum(r['severity_match'] == 'True' for r in output_rows)}/{len(output_rows)}")
    print(f"  guard applied: {sum(str(r['final_report_guard_applied']).lower() == 'true' for r in output_rows)}/{len(output_rows)}")


if __name__ == "__main__":
    main()
