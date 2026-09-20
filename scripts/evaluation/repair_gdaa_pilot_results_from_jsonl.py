from pathlib import Path
import csv
import json
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.csv"
JSONL_PATH = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.jsonl"

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

def norm(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

def severity_id_from_label(label, scenario):
    n = norm(label)

    if scenario == "wildfire_building":
        for key, value in WILDFIRE_CLASS_MAP.items():
            if n == key or n.startswith(key):
                return value

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

def extract_pred(result, scenario):
    if not isinstance(result, dict):
        return "", ""

    candidates = [
        result.get("overall_severity"),
        result.get("severity_label"),
        result.get("predicted_severity"),
        result.get("damage_severity"),
        result.get("severity"),
        result.get("damage_level"),
        result.get("wildfire_damage_class"),
        result.get("damage_class"),
        result.get("class_label"),
        result.get("predicted_class"),
    ]

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
            candidates.extend([
                sub.get("overall_severity"),
                sub.get("severity_label"),
                sub.get("predicted_severity"),
                sub.get("damage_severity"),
                sub.get("severity"),
                sub.get("damage_level"),
                sub.get("wildfire_damage_class"),
                sub.get("damage_class"),
                sub.get("class_label"),
                sub.get("predicted_class"),
            ])

    for item in candidates:
        if item is not None and str(item).strip():
            label = str(item).strip()
            return label, severity_id_from_label(label, scenario)

    return "", ""

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)
    fieldnames = reader.fieldnames

raw_by_case = OrderedDict()

with JSONL_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        payload = json.loads(line)
        raw_by_case[payload["case_id"]] = payload

updated = 0

for row in csv_rows:
    case_id = row["case_id"]
    payload = raw_by_case.get(case_id)

    if not payload:
        continue

    pred_label, pred_id = extract_pred(payload.get("result", {}), row["scenario"])

    if pred_label and str(row.get("pred_severity_id", "")).strip() == "":
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
    writer.writerows(csv_rows)

print(f"CSV repaired: {CSV_PATH}")
print(f"Rows updated from JSONL: {updated}")

print("\nCurrent rows:")
for row in csv_rows:
    print(
        f"{row['case_id']} | scenario={row['scenario']} | "
        f"gt={row['gt_severity_label']} ({row['gt_severity_id']}) | "
        f"pred={row['pred_severity_label']} ({row['pred_severity_id']}) | "
        f"correct={row['severity_correct']}"
    )
