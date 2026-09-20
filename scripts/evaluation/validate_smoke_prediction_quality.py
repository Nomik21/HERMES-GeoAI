from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_RESULTS_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_run_results.csv"
QUALITY_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_prediction_quality.csv"
INVALID_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_invalid_cases.csv"

FIELDNAMES = [
    "case_id",
    "scenario",
    "gt_severity_label",
    "status",
    "report_exists",
    "prediction_valid",
    "invalid_reason",
    "analysis_mode",
    "severity",
    "overall_severity",
    "three_level_severity",
    "wildfire_damage_class",
    "confidence",
    "llm_error",
    "severity_rationale",
    "report_json_path",
]


def read_json(path: Path):
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def get_damage_assessment(report):
    da = report.get("damage_assessment")
    return da if isinstance(da, dict) else {}


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
        "error",
        "exception",
        "traceback",
    ]
    return any(term in text for term in error_terms)


with RUN_RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    run_rows = list(csv.DictReader(f))

quality_rows = []
invalid_rows = []

for row in run_rows:
    report_path = PROJECT_ROOT / row["report_json_path"]
    report = read_json(report_path)
    da = get_damage_assessment(report)

    scenario = row["scenario"]

    severity = clean(da.get("severity"))
    overall_severity = clean(da.get("overall_severity"))
    three_level_severity = clean(da.get("three_level_severity"))
    wildfire_damage_class = clean(da.get("wildfire_damage_class") or da.get("domain_specific_class"))
    confidence = clean(da.get("confidence"))
    llm_error = clean(report.get("llm_error"))
    severity_rationale = clean(report.get("severity_rationale"))
    analysis_mode = clean(report.get("report_scenario") or da.get("analysis_mode"))

    invalid_reason = ""

    if row.get("status") != "success":
        invalid_reason = f"execution_status_{row.get('status')}"
    elif row.get("report_exists") != "True":
        invalid_reason = "missing_report"
    elif contains_error_text(llm_error, severity_rationale):
        invalid_reason = "model_or_quota_error"
    elif scenario == "wildfire_building" and is_unknown(wildfire_damage_class):
        invalid_reason = "missing_wildfire_damage_class"
    elif scenario in {"cross_view_svi_rsi", "pre_post_svi"} and is_unknown(overall_severity) and is_unknown(severity) and is_unknown(three_level_severity):
        invalid_reason = "missing_severity"
    elif confidence in {"0", "0.0", ""}:
        invalid_reason = "zero_or_missing_confidence"

    prediction_valid = invalid_reason == ""

    out = {
        "case_id": row["case_id"],
        "scenario": scenario,
        "gt_severity_label": row["gt_severity_label"],
        "status": row["status"],
        "report_exists": row["report_exists"],
        "prediction_valid": str(prediction_valid),
        "invalid_reason": invalid_reason,
        "analysis_mode": analysis_mode,
        "severity": severity,
        "overall_severity": overall_severity,
        "three_level_severity": three_level_severity,
        "wildfire_damage_class": wildfire_damage_class,
        "confidence": confidence,
        "llm_error": llm_error,
        "severity_rationale": severity_rationale,
        "report_json_path": row["report_json_path"],
    }

    quality_rows.append(out)

    if not prediction_valid:
        invalid_rows.append(row)

with QUALITY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(quality_rows)

with INVALID_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=run_rows[0].keys())
    writer.writeheader()
    writer.writerows(invalid_rows)

print(f"Quality CSV written to: {QUALITY_CSV}")
print(f"Invalid cases CSV written to: {INVALID_CSV}")

print(f"\nTotal cases: {len(quality_rows)}")
print(f"Valid predictions: {sum(r['prediction_valid'] == 'True' for r in quality_rows)}")
print(f"Invalid predictions: {sum(r['prediction_valid'] == 'False' for r in quality_rows)}")

print("\nInvalid cases by reason:")
reasons = {}
for r in quality_rows:
    if r["prediction_valid"] == "False":
        reasons[r["invalid_reason"]] = reasons.get(r["invalid_reason"], 0) + 1

for reason, count in reasons.items():
    print(f"  {reason}: {count}")

print("\nInvalid cases:")
for r in quality_rows:
    if r["prediction_valid"] == "False":
        print(f"  {r['scenario']} | {r['case_id']} | {r['invalid_reason']}")
