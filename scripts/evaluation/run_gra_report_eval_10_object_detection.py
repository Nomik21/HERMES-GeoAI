from pathlib import Path
import csv
import subprocess
import sys
import os
import time
import json
import shutil
import re
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set" / "gra_report_eval_40_manifest.csv"
RESULTS_CSV = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set" / "gra_report_eval_10_object_detection_run_results.csv"
OUT_ROOT = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set"
LOG_DIR = PROJECT_ROOT / "evaluation" / "logs" / "gra_report_eval_10_object_detection"

FIELDNAMES = [
    "run_timestamp",
    "case_id",
    "dataset",
    "scenario",
    "image_paths",
    "source_types",
    "hazard_type",
    "pred_mode",
    "status",
    "returncode",
    "elapsed_seconds",
    "report_json_path",
    "report_md_path",
    "report_html_path",
    "report_exists",
    "error_summary",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "case"


def expected_report_paths(case_id):
    report_dir = PROJECT_ROOT / "outputs" / "Scenarios" / "object_detection" / "reports"

    return {
        "json": report_dir / f"object_detection__{case_id}_report.json",
        "md": report_dir / f"object_detection__{case_id}_report.md",
        "html": report_dir / f"object_detection__{case_id}_report.html",
    }


def copy_reports(case_id, paths):
    dst_dir = OUT_ROOT / "object_detection" / "reports" / safe_name(case_id)
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in [paths["json"], paths["md"], paths["html"]]:
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
            copied += 1

    return copied


def main():
    rows = read_csv(MANIFEST)
    rows = [r for r in rows if r.get("scenario") == "object_detection"]

    existing = []
    if RESULTS_CSV.exists():
        existing = read_csv(RESULTS_CSV)

    successful = {
        r["case_id"]
        for r in existing
        if r.get("status") == "success" and r.get("report_exists") == "True"
    }

    results = existing[:]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("HERMES object_detection report run")
    print("=" * 80)
    print(f"Manifest: {MANIFEST}")
    print(f"Object-detection cases: {len(rows)}")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Log dir: {LOG_DIR}")
    print("")

    for idx, row in enumerate(rows, start=1):
        case_id = row["case_id"]

        if case_id in successful:
            print(f"[{idx}/{len(rows)}] SKIP success: {case_id}")
            continue

        image_paths = row["image_paths"]
        source_types = row.get("source_types") or "SV"
        hazard_type = row.get("gt_hazard_type") or "unknown"
        pred_mode = row.get("gt_pred_mode") or "post_sv_single"

        report_paths = expected_report_paths(case_id)

        # Clean possible old outputs for this case.
        for p in report_paths.values():
            if p.exists():
                p.unlink()

        cmd = [
            sys.executable,
            "-m",
            "hermes.main",
            image_paths,
            source_types,
            hazard_type,
            "object_detection",
            pred_mode,
            "true",
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        safe_case = safe_name(case_id)
        stdout_log = LOG_DIR / f"object_detection__{safe_case}.stdout.txt"
        stderr_log = LOG_DIR / f"object_detection__{safe_case}.stderr.txt"

        print(f"\n[{idx}/{len(rows)}] RUN object_detection | {case_id}")
        print(f"  image_paths: {image_paths}")
        print(f"  hazard_type: {hazard_type}")

        start = time.time()

        try:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
                env=env,
            )

            elapsed = time.time() - start

            stdout_log.write_text(proc.stdout or "", encoding="utf-8")
            stderr_log.write_text(proc.stderr or "", encoding="utf-8")

            report_exists = report_paths["json"].exists()

            if proc.returncode != 0:
                status = "failed"
                error_summary = (proc.stderr or proc.stdout or "")[-700:].replace("\n", " ")
            elif not report_exists:
                status = "missing_report"
                error_summary = "Process returned 0 but expected object_detection report JSON was not found."
            else:
                status = "success"
                error_summary = ""

            copied = copy_reports(case_id, report_paths) if report_exists else 0

            print(
                f"  status={status} | returncode={proc.returncode} | "
                f"elapsed={elapsed:.2f}s | report_exists={report_exists} | copied_files={copied}"
            )

            results.append({
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "dataset": row.get("dataset", ""),
                "scenario": "object_detection",
                "image_paths": image_paths,
                "source_types": source_types,
                "hazard_type": hazard_type,
                "pred_mode": pred_mode,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": f"{elapsed:.2f}",
                "report_json_path": str(report_paths["json"].relative_to(PROJECT_ROOT)),
                "report_md_path": str(report_paths["md"].relative_to(PROJECT_ROOT)),
                "report_html_path": str(report_paths["html"].relative_to(PROJECT_ROOT)),
                "report_exists": str(bool(report_exists)),
                "error_summary": error_summary,
            })

            write_csv(RESULTS_CSV, results)

        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - start
            stdout_log.write_text(exc.stdout or "", encoding="utf-8")
            stderr_log.write_text(exc.stderr or "", encoding="utf-8")

            print(f"  status=timeout | elapsed={elapsed:.2f}s")

            results.append({
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_id": case_id,
                "dataset": row.get("dataset", ""),
                "scenario": "object_detection",
                "image_paths": image_paths,
                "source_types": source_types,
                "hazard_type": hazard_type,
                "pred_mode": pred_mode,
                "status": "timeout",
                "returncode": "timeout",
                "elapsed_seconds": f"{elapsed:.2f}",
                "report_json_path": str(report_paths["json"].relative_to(PROJECT_ROOT)),
                "report_md_path": str(report_paths["md"].relative_to(PROJECT_ROOT)),
                "report_html_path": str(report_paths["html"].relative_to(PROJECT_ROOT)),
                "report_exists": "False",
                "error_summary": "Timeout after 1800 seconds.",
            })

            write_csv(RESULTS_CSV, results)

    print("")
    print("Done.")
    print(f"Results written to: {RESULTS_CSV}")
    print(f"Reports copied under: {OUT_ROOT / 'object_detection' / 'reports'}")


if __name__ == "__main__":
    main()
