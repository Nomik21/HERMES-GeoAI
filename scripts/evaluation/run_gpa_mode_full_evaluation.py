from pathlib import Path
import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from collections import OrderedDict

from hermes.core.geo_perception import ModePerceiver, ensure_pil_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_results.csv"

ALLOWED_MODES = [
    "post_sv_single",
    "post_rs_single",
    "post_paired_sv_rs",
    "bitemporal_sv",
    "bitemporal_rs",
]

FIELDNAMES = [
    "run_timestamp",
    "case_id",
    "original_case_id",
    "dataset",
    "scenario",
    "image_paths",
    "original_image_paths",
    "source_types",
    "num_images",
    "gt_pred_mode",
    "pred_mode",
    "pred_mode_correct",
    "mode_confidence",
    "gt_hazard_type",
    "pred_hazard_type",
    "hazard_confidence",
    "needs_restoration",
    "status",
    "elapsed_seconds",
    "error_summary",
]


def resolve_project_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def split_paths(path_text):
    return [p.strip() for p in str(path_text or "").split(";") if p.strip()]


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

    existing_rows = load_existing(output_csv) if args.resume else []
    latest_by_case = OrderedDict()
    for row in existing_rows:
        latest_by_case[row["case_id"]] = row

    completed_success = {
        case_id
        for case_id, row in latest_by_case.items()
        if row.get("status") == "success"
    }

    output_rows = existing_rows[:]

    mode_perceiver = ModePerceiver()

    print(f"Manifest: {manifest_path}")
    print(f"Target GPA mode cases: {len(manifest_rows)}")
    print(f"Output CSV: {output_csv}")
    print(f"Resume: {args.resume}")
    print(f"Max new executions: {args.max_new}")

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

        print(
            f"\n[{idx}/{len(manifest_rows)}] RUN ModePerceiver | "
            f"{case_id} | gt_mode={row['gt_pred_mode']} | scenario={row['scenario']}"
        )

        start = time.time()

        try:
            image_paths = split_paths(row["image_paths"])
            images = [ensure_pil_image(str(resolve_project_path(p))) for p in image_paths]

            result = mode_perceiver.infer(images)

            elapsed = time.time() - start

            pred_mode = norm(result.get("pred_mode"))
            gt_mode = norm(row.get("gt_pred_mode"))

            if pred_mode not in ALLOWED_MODES:
                status = "schema_invalid"
                error_summary = f"pred_mode outside allowed modes: {pred_mode}"
            else:
                status = "success"
                error_summary = ""

            out = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "original_case_id": row["original_case_id"],
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                "image_paths": row["image_paths"],
                "original_image_paths": row["original_image_paths"],
                "source_types": row["source_types"],
                "num_images": row["num_images"],
                "gt_pred_mode": gt_mode,
                "pred_mode": pred_mode,
                "pred_mode_correct": str(gt_mode == pred_mode),
                "mode_confidence": result.get("mode_confidence", ""),
                "gt_hazard_type": row["gt_hazard_type"],
                "pred_hazard_type": result.get("hazard_type", ""),
                "hazard_confidence": result.get("hazard_confidence", ""),
                "needs_restoration": json.dumps(result.get("needs_restoration", ""), ensure_ascii=False),
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                "error_summary": error_summary,
            }

        except Exception:
            elapsed = time.time() - start

            out = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "original_case_id": row.get("original_case_id", ""),
                "dataset": row.get("dataset", ""),
                "scenario": row.get("scenario", ""),
                "image_paths": row.get("image_paths", ""),
                "original_image_paths": row.get("original_image_paths", ""),
                "source_types": row.get("source_types", ""),
                "num_images": row.get("num_images", ""),
                "gt_pred_mode": norm(row.get("gt_pred_mode")),
                "pred_mode": "",
                "pred_mode_correct": "False",
                "mode_confidence": "",
                "gt_hazard_type": row.get("gt_hazard_type", ""),
                "pred_hazard_type": "",
                "hazard_confidence": "",
                "needs_restoration": "",
                "status": "failed",
                "elapsed_seconds": round(elapsed, 2),
                "error_summary": traceback.format_exc()[-1200:].replace("\n", " "),
            }

        output_rows.append(out)
        write_rows(output_csv, output_rows)

        print(
            f"  status={out['status']} | "
            f"pred_mode={out['pred_mode']} | "
            f"mode_correct={out['pred_mode_correct']} | "
            f"pred_hazard={out['pred_hazard_type']} | "
            f"elapsed={out['elapsed_seconds']}s"
        )

    print("\nDone.")
    print(f"New executions this run: {executed}")
    print(f"Results written to: {output_csv}")


if __name__ == "__main__":
    main()
