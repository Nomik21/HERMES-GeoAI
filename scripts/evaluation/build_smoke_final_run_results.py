from pathlib import Path
import csv
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SMOKE_MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest_smoke.csv"
OUTPUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_final_run_results.csv"

SUPPORTED_SCENARIOS = {
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
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


def case_part_from_image_paths(image_paths):
    stems = []
    for rel_path in image_paths.split(";"):
        rel_path = rel_path.strip()
        if rel_path:
            stems.append(Path(rel_path).stem)
    return "__".join(stems)


def expected_report_json(row):
    scenario = row["scenario"]
    case_part = case_part_from_image_paths(row["image_paths"])
    return (
        PROJECT_ROOT
        / "outputs"
        / "Scenarios"
        / scenario
        / "reports"
        / f"{scenario}__{case_part}_report.json"
    )


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


with SMOKE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    manifest_rows = list(csv.DictReader(f))

rows = []

for row in manifest_rows:
    if row["scenario"] not in SUPPORTED_SCENARIOS:
        continue

    report_path = expected_report_json(row)
    report_exists = report_path.exists()

    if report_exists:
        prediction_valid, invalid_reason = validate_prediction(row, report_path)
    else:
        prediction_valid = False
        invalid_reason = "missing_report"

    status = "success" if prediction_valid else "invalid_prediction"

    rows.append({
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_id": row["case_id"],
        "dataset": row["dataset"],
        "scenario": row["scenario"],
        "gt_hazard_type": row["gt_hazard_type"],
        "gt_severity_label": row["gt_severity_label"],
        "gt_severity_id": row["gt_severity_id"],
        "gt_pred_mode": row["gt_pred_mode"],
        "image_paths": row["image_paths"],
        "source_types": row["source_types"],
        "status": status,
        "returncode": "",
        "elapsed_seconds": "",
        "report_json_path": str(report_path.relative_to(PROJECT_ROOT)),
        "report_exists": str(report_exists),
        "prediction_valid": str(prediction_valid),
        "invalid_reason": invalid_reason,
        "stdout_log": "",
        "stderr_log": "",
        "error_summary": invalid_reason,
    })

with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Final smoke run results written to: {OUTPUT_CSV}")
print(f"Rows written: {len(rows)}")

valid = sum(r["prediction_valid"] == "True" for r in rows)
invalid = len(rows) - valid

print(f"\nValid predictions: {valid}/{len(rows)}")
print(f"Invalid predictions: {invalid}/{len(rows)}")

print("\nRows by scenario/status:")
counts = {}
for r in rows:
    key = (r["scenario"], r["status"])
    counts[key] = counts.get(key, 0) + 1

for key, count in counts.items():
    print(f"  {key}: {count}")

if invalid:
    print("\nInvalid rows:")
    for r in rows:
        if r["prediction_valid"] != "True":
            print(f"  {r['scenario']} | {r['case_id']} | {r['invalid_reason']}")
else:
    print("\nFINAL SMOKE VALIDATION PASSED ✅")
