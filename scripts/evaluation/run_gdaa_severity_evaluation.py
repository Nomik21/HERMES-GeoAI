from pathlib import Path
import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from collections import OrderedDict

from hermes.core.geo_damage_analysis import analyze_damage

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.csv"
DEFAULT_JSONL = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.jsonl"

FIELDNAMES = [
    "run_timestamp",
    "case_id",
    "dataset",
    "scenario",
    "image_paths",
    "source_types",
    "gt_hazard_type",
    "gt_severity_label",
    "gt_severity_id",
    "gt_pred_mode",
    "pred_severity_label",
    "pred_severity_id",
    "severity_correct",
    "status",
    "elapsed_seconds",
    "model_name",
    "error_summary",
]


THREE_CLASS_MAP = {
    "mild": 0,
    "minor": 0,
    "moderate": 1,
    "severe": 2,
}

WILDFIRE_CLASS_MAP = {
    "0_no_damage": 0,
    "no_damage": 0,
    "0": 0,

    "1_affected_1_9": 1,
    "affected_1_9": 1,
    "affected": 1,
    "1": 1,

    "2_minor_10_25": 2,
    "minor_10_25": 2,
    "minor": 2,
    "2": 2,

    "3_major_26_50": 3,
    "major_26_50": 3,
    "major": 3,
    "3": 3,

    "4_destroyed_50plus": 4,
    "destroyed_50plus": 4,
    "destroyed": 4,
    "4": 4,
}


def resolve_project_path(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def split_semicolon(value):
    return [p.strip() for p in str(value or "").split(";") if p.strip()]


def norm(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def load_existing(output_csv):
    if not output_csv.exists():
        return []

    with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(output_csv, rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(jsonl_path, payload):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_pred_severity(result, scenario):
    """
    Robust extraction from HERMES GDAA schemas.

    For wildfire_building, the evaluation target is the canonical
    five-class wildfire_damage_class, not the derived overall_severity.
    """
    if not isinstance(result, dict):
        return "", ""

    candidates = []

    if scenario == "wildfire_building":
        candidates.extend([
            result.get("wildfire_damage_class"),
            result.get("Predicted_Class"),
            result.get("predicted_class"),
            result.get("damage_class"),
            result.get("class_label"),
            result.get("overall_severity"),
        ])
    else:
        candidates.extend([
            result.get("overall_severity"),
            result.get("Predicted_Severity"),
            result.get("predicted_severity"),
            result.get("severity_label"),
            result.get("damage_severity"),
            result.get("severity"),
            result.get("damage_level"),
            result.get("damage_class"),
            result.get("class_label"),
            result.get("predicted_class"),
        ])

    for key in [
        "classification",
        "severity_assessment",
        "damage_assessment",
        "building_damage",
        "summary",
        "result",
        "gdaa_output",
    ]:
        sub = result.get(key)
        if isinstance(sub, dict):
            if scenario == "wildfire_building":
                candidates.extend([
                    sub.get("wildfire_damage_class"),
                    sub.get("Predicted_Class"),
                    sub.get("predicted_class"),
                    sub.get("damage_class"),
                    sub.get("class_label"),
                    sub.get("overall_severity"),
                ])
            else:
                candidates.extend([
                    sub.get("overall_severity"),
                    sub.get("Predicted_Severity"),
                    sub.get("predicted_severity"),
                    sub.get("severity_label"),
                    sub.get("damage_severity"),
                    sub.get("severity"),
                    sub.get("damage_level"),
                    sub.get("damage_class"),
                    sub.get("class_label"),
                    sub.get("predicted_class"),
                ])

    for item in candidates:
        if item is not None and str(item).strip():
            label = str(item).strip()

            # Avoid treating derived wildfire "None" as an unrecognized class
            # when a canonical wildfire_damage_class is absent.
            if scenario == "wildfire_building" and label.lower() == "none":
                return "0_No_Damage", 0

            return label, severity_id_from_label(label, scenario)

    return "", ""


def severity_id_from_label(label, scenario):
    n = norm(label)

    if scenario == "wildfire_building":
        for key, value in WILDFIRE_CLASS_MAP.items():
            if n == key or n.startswith(key):
                return value

        # Some outputs contain readable strings.
        if "no_damage" in n:
            return 0
        if "affected" in n or "1_9" in n:
            return 1
        if "minor" in n or "10_25" in n:
            return 2
        if "major" in n or "26_50" in n:
            return 3
        if "destroyed" in n or "50" in n:
            return 4

        return ""

    for key, value in THREE_CLASS_MAP.items():
        if n == key or key in n:
            return value

    return ""


def normalize_gt_id(row):
    raw = str(row.get("gt_severity_id", "")).strip()
    if raw != "":
        try:
            return int(float(raw))
        except ValueError:
            pass

    return severity_id_from_label(row.get("gt_severity_label", ""), row.get("scenario", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    output_csv = resolve_project_path(args.output)
    jsonl_path = resolve_project_path(args.jsonl)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    existing_rows = load_existing(output_csv) if args.resume else []
    latest_by_case = OrderedDict()
    for row in existing_rows:
        latest_by_case[row["case_id"]] = row

    completed_success = {
        case_id
        for case_id, row in latest_by_case.items()
        if row.get("status") == "success" and str(row.get("pred_severity_id", "")).strip() != ""
    }

    output_rows = existing_rows[:]

    print(f"Manifest: {manifest_path}")
    print(f"Target GDAA severity cases: {len(manifest_rows)}")
    print(f"Output CSV: {output_csv}")
    print(f"JSONL: {jsonl_path}")
    print(f"Resume: {args.resume}")
    print(f"Max new executions: {args.max_new}")
    print(f"Model override: {args.model_name}")

    executed = 0

    for idx, row in enumerate(manifest_rows, start=1):
        case_id = row["case_id"]

        if args.resume and case_id in completed_success:
            print(f"[{idx}/{len(manifest_rows)}] SKIP success: {case_id}")
            continue

        if args.max_new is not None and executed >= args.max_new:
            print(f"\nReached --max-new={args.max_new}. Stopping.")
            break

        executed += 1

        scenario = row["scenario"]
        image_paths = split_semicolon(row["image_paths"])
        source_types = split_semicolon(row["source_types"])
        resolved_image_paths = [str(resolve_project_path(p)) for p in image_paths]

        print(
            f"\n[{idx}/{len(manifest_rows)}] RUN GDAA | "
            f"{case_id} | scenario={scenario} | gt={row['gt_severity_label']}"
        )

        start = time.time()

        try:
            result = analyze_damage(
                image_paths=resolved_image_paths,
                source_types=source_types,
                hazard_type=row["gt_hazard_type"],
                pred_mode=row["gt_pred_mode"],
                analysis_mode=scenario,
                pair_id=case_id,
                model_name=args.model_name,
                visualize=False,
            )

            elapsed = time.time() - start

            pred_label, pred_id = extract_pred_severity(result, scenario)
            gt_id = normalize_gt_id(row)

            status = "success"
            error_summary = ""

            severity_correct = str(pred_id != "" and int(pred_id) == int(gt_id))

            out = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "dataset": row["dataset"],
                "scenario": scenario,
                "image_paths": row["image_paths"],
                "source_types": row["source_types"],
                "gt_hazard_type": row["gt_hazard_type"],
                "gt_severity_label": row["gt_severity_label"],
                "gt_severity_id": gt_id,
                "gt_pred_mode": row["gt_pred_mode"],
                "pred_severity_label": pred_label,
                "pred_severity_id": pred_id,
                "severity_correct": severity_correct,
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                "model_name": args.model_name or "",
                "error_summary": error_summary,
            }

            append_jsonl(jsonl_path, {
                "case_id": case_id,
                "scenario": scenario,
                "gt_severity_label": row["gt_severity_label"],
                "gt_severity_id": gt_id,
                "pred_severity_label": pred_label,
                "pred_severity_id": pred_id,
                "result": result,
            })

        except Exception:
            elapsed = time.time() - start

            out = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "dataset": row.get("dataset", ""),
                "scenario": row.get("scenario", ""),
                "image_paths": row.get("image_paths", ""),
                "source_types": row.get("source_types", ""),
                "gt_hazard_type": row.get("gt_hazard_type", ""),
                "gt_severity_label": row.get("gt_severity_label", ""),
                "gt_severity_id": row.get("gt_severity_id", ""),
                "gt_pred_mode": row.get("gt_pred_mode", ""),
                "pred_severity_label": "",
                "pred_severity_id": "",
                "severity_correct": "False",
                "status": "failed",
                "elapsed_seconds": round(elapsed, 2),
                "model_name": args.model_name or "",
                "error_summary": traceback.format_exc()[-1500:].replace("\n", " "),
            }

        output_rows.append(out)
        write_rows(output_csv, output_rows)

        print(
            f"  status={out['status']} | "
            f"pred={out['pred_severity_label']} ({out['pred_severity_id']}) | "
            f"correct={out['severity_correct']} | "
            f"elapsed={out['elapsed_seconds']}s"
        )

    print("\nDone.")
    print(f"New executions this run: {executed}")
    print(f"Results written to: {output_csv}")


if __name__ == "__main__":
    main()
