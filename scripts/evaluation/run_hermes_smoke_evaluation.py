from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import threading
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest_smoke.csv"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_run_results.csv"

SUPPORTED_SCENARIOS = {
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
}

CLI_SCENARIO_ARG = {
    "cross_view_svi_rsi": "auto",
    "pre_post_svi": "auto",
    "wildfire_building": "auto",
}

FIELDNAMES = [
    "run_timestamp",
    "case_id",
    "dataset",
    "scenario",
    "gt_hazard_type",
    "gt_severity_label",
    "gt_severity_id",
    "gt_pred_mode",
    "image_paths",
    "source_types",
    "status",
    "returncode",
    "elapsed_seconds",
    "report_json_path",
    "report_exists",
    "prediction_valid",
    "invalid_reason",
    "stdout_log",
    "stderr_log",
    "error_summary",
]


def resolve_project_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def case_part_from_image_paths(image_paths):
    stems = []
    for rel_path in image_paths.split(";"):
        rel_path = rel_path.strip()
        if rel_path:
            stems.append(Path(rel_path).stem)
    return "__".join(stems)


def expected_report_paths(row):
    scenario = row["scenario"]
    case_part = case_part_from_image_paths(row["image_paths"])
    report_dir = PROJECT_ROOT / "outputs" / "Scenarios" / scenario / "reports"

    return {
        "json": report_dir / f"{scenario}__{case_part}_report.json",
        "md": report_dir / f"{scenario}__{case_part}_report.md",
        "html": report_dir / f"{scenario}__{case_part}_report.html",
    }


def expected_intermediate_paths(row):
    case_part = case_part_from_image_paths(row["image_paths"])
    inter_dir = PROJECT_ROOT / "outputs" / "intermediate"

    return [
        inter_dir / f"{case_part}_gpa.json",
        inter_dir / f"{case_part}_gira.json",
        inter_dir / f"{case_part}_gdaa.json",
        inter_dir / f"{case_part}_gra.json",
    ]


def delete_old_outputs(row):
    for path in expected_report_paths(row).values():
        if path.exists():
            path.unlink()

    for path in expected_intermediate_paths(row):
        if path.exists():
            path.unlink()


def read_json(path):
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def is_unknown(value):
    text = clean(value).lower()
    return text in {"", "unknown", "none", "null", "n/a", "not available"}


def contains_error_text(*values):
    text = " ".join(clean(v) for v in values).lower()
    error_terms = [
        "resource_exhausted",
        "quota",
        "429",
        "failed",
        "exception",
        "traceback",
    ]
    return any(term in text for term in error_terms)


def validate_prediction(row, report_path):
    report = read_json(report_path)

    if not report:
        return False, "missing_or_invalid_report_json"

    damage_assessment = report.get("damage_assessment")
    if not isinstance(damage_assessment, dict):
        damage_assessment = {}

    llm_error = clean(report.get("llm_error"))
    severity_rationale = clean(report.get("severity_rationale"))

    if contains_error_text(llm_error, severity_rationale):
        return False, "model_or_quota_error"

    confidence = clean(damage_assessment.get("confidence"))

    scenario = row["scenario"]

    if scenario == "wildfire_building":
        wildfire_damage_class = (
            damage_assessment.get("wildfire_damage_class")
            or damage_assessment.get("domain_specific_class")
            or damage_assessment.get("damage_class")
        )

        if is_unknown(wildfire_damage_class):
            return False, "missing_wildfire_damage_class"

    elif scenario in {"cross_view_svi_rsi", "pre_post_svi"}:
        severity = (
            damage_assessment.get("overall_severity")
            or damage_assessment.get("severity")
            or damage_assessment.get("three_level_severity")
        )

        if is_unknown(severity):
            return False, "missing_severity"

    if confidence in {"", "0", "0.0"}:
        return False, "zero_or_missing_confidence"

    return True, ""


def load_existing_results(results_csv):
    if not results_csv.exists():
        return []

    with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_results(results_csv, rows):
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    with results_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_timeout_stream(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)



def run_subprocess_streaming(cmd, cwd, timeout, env):
    """
    Run subprocess while streaming stdout/stderr live to terminal
    and also returning captured stdout/stderr for log files.
    """
    stdout_chunks = []
    stderr_chunks = []

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    def pump(stream, chunks, writer):
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                writer.write(line)
                writer.flush()
        finally:
            try:
                stream.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=pump,
        args=(proc.stdout, stdout_chunks, sys.stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump,
        args=(proc.stderr, stderr_chunks, sys.stderr),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--clean-old-outputs", action="store_true", default=True)
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    results_csv = resolve_project_path(args.results_csv)

    if args.log_dir:
        log_dir = resolve_project_path(args.log_dir)
    else:
        log_dir = PROJECT_ROOT / "evaluation" / "logs" / results_csv.stem

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    runnable_rows = [
        row for row in manifest_rows
        if row["scenario"] in SUPPORTED_SCENARIOS
    ]

    skipped_rows = [
        row for row in manifest_rows
        if row["scenario"] not in SUPPORTED_SCENARIOS
    ]

    if args.limit is not None:
        runnable_rows = runnable_rows[: args.limit]

    existing_results = load_existing_results(results_csv)

    attempted_keys = {
        (row["case_id"], row["scenario"])
        for row in existing_results
    }

    success_keys = {
        (row["case_id"], row["scenario"])
        for row in existing_results
        if row.get("status") == "success" and row.get("prediction_valid") == "True"
    }

    results = existing_results[:]
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {manifest_path}")
    print(f"Runnable cases in manifest: {len(runnable_rows)}")
    print(f"Skipped unsupported cases: {len(skipped_rows)}")
    print(f"Results CSV: {results_csv}")
    print(f"Log dir: {log_dir}")
    print(f"Resume: {args.resume}")
    print(f"Retry invalid: {args.retry_invalid}")
    print(f"Max new executions: {args.max_new}")

    executed_new = 0

    for index, row in enumerate(runnable_rows, start=1):
        case_id = row["case_id"]
        scenario = row["scenario"]
        key = (case_id, scenario)

        if args.resume:
            if args.retry_invalid:
                if key in success_keys:
                    print(f"[{index}/{len(runnable_rows)}] SKIP valid success: {scenario} | {case_id}")
                    continue
            else:
                if key in attempted_keys:
                    print(f"[{index}/{len(runnable_rows)}] SKIP already attempted: {scenario} | {case_id}")
                    continue

        if args.max_new is not None and executed_new >= args.max_new:
            print(f"\nReached --max-new={args.max_new}. Stopping.")
            break

        executed_new += 1

        print(f"\n[{index}/{len(runnable_rows)}] RUN {scenario} | {case_id}")

        image_paths = row["image_paths"]
        source_types = row["source_types"]
        hazard_type = row["gt_hazard_type"]
        cli_scenario = CLI_SCENARIO_ARG[scenario]
        pred_mode = row["gt_pred_mode"]

        report_paths = expected_report_paths(row)
        report_json_path = report_paths["json"]

        if args.clean_old_outputs:
            delete_old_outputs(row)

        cmd = [
            sys.executable,
            "-m",
            "hermes.main",
            image_paths,
            source_types,
            hazard_type,
            cli_scenario,
            pred_mode,
            "true",
        ]

        safe_case = "".join(c if c.isalnum() or c in "-_." else "_" for c in case_id)
        log_prefix = f"{scenario}__{safe_case}"
        stdout_log = log_dir / f"{log_prefix}.stdout.txt"
        stderr_log = log_dir / f"{log_prefix}.stderr.txt"

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        start = time.time()
        status = "unknown"
        returncode = ""
        error_summary = ""
        prediction_valid = False
        invalid_reason = ""

        try:
            proc = run_subprocess_streaming(
                cmd=cmd,
                cwd=PROJECT_ROOT,
                timeout=args.timeout,
                env=env,
            )

            elapsed = time.time() - start
            returncode = proc.returncode

            stdout_log.write_text(proc.stdout or "", encoding="utf-8")
            stderr_log.write_text(proc.stderr or "", encoding="utf-8")

            report_exists = report_json_path.exists()

            if proc.returncode != 0:
                status = "failed"
                invalid_reason = "process_returncode_nonzero"
                error_summary = (proc.stderr or proc.stdout or "")[-700:].replace("\n", " ")

            elif not report_exists:
                status = "missing_report"
                invalid_reason = "missing_report"
                error_summary = "Process returned 0 but expected report JSON was not found."

            else:
                prediction_valid, invalid_reason = validate_prediction(row, report_json_path)

                if prediction_valid:
                    status = "success"
                else:
                    status = "invalid_prediction"
                    error_summary = invalid_reason

        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - start
            status = "timeout"
            returncode = "timeout"
            report_exists = report_json_path.exists()
            prediction_valid = False
            invalid_reason = "timeout"
            error_summary = f"Timeout after {args.timeout} seconds."

            stdout_log.write_text(normalize_timeout_stream(exc.stdout), encoding="utf-8")
            stderr_log.write_text(normalize_timeout_stream(exc.stderr), encoding="utf-8")

        result = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "case_id": case_id,
            "dataset": row["dataset"],
            "scenario": scenario,
            "gt_hazard_type": row["gt_hazard_type"],
            "gt_severity_label": row["gt_severity_label"],
            "gt_severity_id": row["gt_severity_id"],
            "gt_pred_mode": row["gt_pred_mode"],
            "image_paths": row["image_paths"],
            "source_types": row["source_types"],
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 2),
            "report_json_path": str(report_json_path.relative_to(PROJECT_ROOT)),
            "report_exists": str(bool(report_exists)),
            "prediction_valid": str(bool(prediction_valid)),
            "invalid_reason": invalid_reason,
            "stdout_log": str(stdout_log.relative_to(PROJECT_ROOT)),
            "stderr_log": str(stderr_log.relative_to(PROJECT_ROOT)),
            "error_summary": error_summary,
        }

        results.append(result)
        write_results(results_csv, results)

        print(
            f"  status={status} | "
            f"prediction_valid={prediction_valid} | "
            f"reason={invalid_reason or '-'} | "
            f"returncode={returncode} | "
            f"elapsed={round(elapsed, 2)}s | "
            f"report_exists={report_exists}"
        )

    print("\nDone.")
    print(f"New executions this run: {executed_new}")
    print(f"Results written to: {results_csv}")


if __name__ == "__main__":
    main()
