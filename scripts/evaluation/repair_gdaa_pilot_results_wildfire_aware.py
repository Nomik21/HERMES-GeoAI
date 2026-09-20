from pathlib import Path
import csv
import json
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.csv"
JSONL_PATH = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.jsonl"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "evaluation" / "run_gdaa_severity_evaluation.py"

spec = importlib.util.spec_from_file_location("runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

raw_by_case = {}

with JSONL_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        payload = json.loads(line)
        raw_by_case[payload["case_id"]] = payload

updated = 0

for row in rows:
    payload = raw_by_case.get(row["case_id"])
    if not payload:
        continue

    pred_label, pred_id = runner.extract_pred_severity(
        payload.get("result", {}),
        row["scenario"],
    )

    row["pred_severity_label"] = pred_label
    row["pred_severity_id"] = pred_id

    try:
        row["severity_correct"] = str(int(pred_id) == int(row["gt_severity_id"]))
    except Exception:
        row["severity_correct"] = "False"

    updated += 1

with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Repaired CSV: {CSV_PATH}")
print(f"Rows updated: {updated}")

print("\nPilot rows after wildfire-aware repair:")
for row in rows:
    print(
        f"{row['case_id']} | {row['scenario']} | "
        f"gt={row['gt_severity_label']} ({row['gt_severity_id']}) | "
        f"pred={row['pred_severity_label']} ({row['pred_severity_id']}) | "
        f"correct={row['severity_correct']}"
    )
