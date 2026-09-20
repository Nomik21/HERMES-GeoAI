from pathlib import Path
import argparse
import csv
import json
import time
import traceback
from datetime import datetime

from hermes.tools.perception_tools import geo_perception_tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest_smoke.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "hermes_gpa_hazard_smoke_results.csv"

TARGET_SCENARIO = "single_svi_hazard_classification"

FIELDNAMES = [
    "run_timestamp",
    "case_id",
    "dataset",
    "scenario",
    "image_paths",
    "source_types",
    "gt_hazard_type",
    "pred_hazard_type",
    "hazard_correct",
    "gt_pred_mode",
    "pred_mode",
    "pred_mode_correct",
    "hazard_confidence",
    "mode_confidence",
    "needs_restoration",
    "status",
    "elapsed_seconds",
    "gpa_intermediate_path",
    "error_summary",
]


def resolve_project_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        "other": "other",
        "unknown": "unknown",
    }

    return aliases.get(text, text)


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


def extract_prediction(result):
    mode_and_hazard = result.get("mode_and_hazard", {}) if isinstance(result, dict) else {}

    return {
        "pred_hazard_type": mode_and_hazard.get("hazard_type", ""),
        "hazard_confidence": mode_and_hazard.get("hazard_confidence", ""),
        "pred_mode": mode_and_hazard.get("pred_mode", ""),
        "mode_confidence": mode_and_hazard.get("mode_confidence", ""),
        "needs_restoration": json.dumps(mode_and_hazard.get("needs_restoration", ""), ensure_ascii=False),
        "gpa_intermediate_path": result.get("_hermes_intermediate_path", "") if isinstance(result, dict) else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    output_csv = resolve_project_path(args.output)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    target_rows = [
        row for row in manifest_rows
        if row["scenario"] == TARGET_SCENARIO
    ]

    existing_rows = load_existing(output_csv)
    completed_keys = {
        row["case_id"]
        for row in existing_rows
        if row.get("status") == "success"
    }

    output_rows = existing_rows[:]

    print(f"Manifest: {manifest_path}")
    print(f"Target GPA hazard cases: {len(target_rows)}")
    print(f"Output CSV: {output_csv}")
    print(f"Resume: {args.resume}")
    print(f"Max new executions: {args.max_new}")

    executed = 0

    for idx, row in enumerate(target_rows, start=1):
        case_id = row["case_id"]

        if args.resume and case_id in completed_keys:
            print(f"[{idx}/{len(target_rows)}] SKIP success: {case_id}")
            continue

        if args.max_new is not None and executed >= args.max_new:
            print(f"\nReached --max-new={args.max_new}. Stopping.")
            break

        executed += 1

        print(f"\n[{idx}/{len(target_rows)}] RUN GPA | {case_id} | gt={row['gt_hazard_type']}")

        start = time.time()
        status = "unknown"
        error_summary = ""

        try:
            result = geo_perception_tool.func(image_paths=row["image_paths"])
            elapsed = time.time() - start

            pred = extract_prediction(result)

            gt_hazard = normalize_hazard(row["gt_hazard_type"])
            pred_hazard = normalize_hazard(pred["pred_hazard_type"])

            gt_mode = row["gt_pred_mode"]
            pred_mode = pred["pred_mode"]

            status = "success"

            out = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                "image_paths": row["image_paths"],
                "source_types": row["source_types"],
                "gt_hazard_type": gt_hazard,
                "pred_hazard_type": pred_hazard,
                "hazard_correct": str(gt_hazard == pred_hazard),
                "gt_pred_mode": gt_mode,
                "pred_mode": pred_mode,
                "pred_mode_correct": str(gt_mode == pred_mode),
                "hazard_confidence": pred["hazard_confidence"],
                "mode_confidence": pred["mode_confidence"],
                "needs_restoration": pred["needs_restoration"],
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                "gpa_intermediate_path": pred["gpa_intermediate_path"],
                "error_summary": "",
            }

        except Exception:
            elapsed = time.time() - start
            status = "failed"
            error_summary = traceback.format_exc()[-1000:].replace("\n", " ")

            out = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                "image_paths": row["image_paths"],
                "source_types": row["source_types"],
                "gt_hazard_type": normalize_hazard(row["gt_hazard_type"]),
                "pred_hazard_type": "",
                "hazard_correct": "False",
                "gt_pred_mode": row["gt_pred_mode"],
                "pred_mode": "",
                "pred_mode_correct": "False",
                "hazard_confidence": "",
                "mode_confidence": "",
                "needs_restoration": "",
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                "gpa_intermediate_path": "",
                "error_summary": error_summary,
            }

        output_rows.append(out)
        write_rows(output_csv, output_rows)

        print(
            f"  status={status} | "
            f"pred_hazard={out['pred_hazard_type']} | "
            f"hazard_correct={out['hazard_correct']} | "
            f"pred_mode={out['pred_mode']} | "
            f"mode_correct={out['pred_mode_correct']} | "
            f"elapsed={out['elapsed_seconds']}s"
        )

    print("\nDone.")
    print(f"New executions this run: {executed}")
    print(f"Results written to: {output_csv}")


if __name__ == "__main__":
    main()
