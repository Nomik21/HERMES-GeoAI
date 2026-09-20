from pathlib import Path
import csv
import json
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRED_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_final_predictions.csv"
SUMMARY_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_metrics_summary.csv"
CONFUSION_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_confusion_matrix.csv"
SUMMARY_TXT = PROJECT_ROOT / "evaluation" / "hermes_smoke_metrics_summary.txt"

THREE_CLASS_ORDER = {
    "Mild": 0,
    "Moderate": 1,
    "Severe": 2,
}

WILDFIRE_ORDER = {
    "0_No_Damage": 0,
    "1_Affected_1_9": 1,
    "2_Minor_10_25": 2,
    "3_Major_26_50": 3,
    "4_Destroyed_50plus": 4,
}


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_three_class(value):
    text = clean(value)

    mapping = {
        "minor": "Mild",
        "mild": "Mild",
        "moderate": "Moderate",
        "major": "Severe",
        "severe": "Severe",
        "destroyed": "Severe",
    }

    return mapping.get(text.lower(), text)


def wildfire_class_from_value(value):
    text = clean(value)

    if text in WILDFIRE_ORDER:
        return text

    # fallback from severity-like labels
    lower = text.lower()

    if lower in {"none", "no damage", "no_damage"}:
        return "0_No_Damage"
    if lower in {"affected"}:
        return "1_Affected_1_9"
    if lower in {"minor", "mild"}:
        return "2_Minor_10_25"
    if lower in {"major"}:
        return "3_Major_26_50"
    if lower in {"destroyed", "severe"}:
        return "4_Destroyed_50plus"

    return text


def get_gt_pred_label(row):
    scenario = row["scenario"]

    if scenario == "wildfire_building":
        return row["gt_severity_label"]

    return normalize_three_class(row["gt_severity_label"])


def get_pred_label(row):
    scenario = row["scenario"]

    if scenario == "wildfire_building":
        return wildfire_class_from_value(row["pred_damage_class"])

    return normalize_three_class(row["pred_severity_label"])


def get_class_order(scenario):
    if scenario == "wildfire_building":
        return WILDFIRE_ORDER
    return THREE_CLASS_ORDER


def exact_accuracy(rows):
    if not rows:
        return 0.0

    correct = sum(get_gt_pred_label(r) == get_pred_label(r) for r in rows)
    return correct / len(rows)


def mean_ncse(rows):
    values = []

    for row in rows:
        scenario = row["scenario"]
        order = get_class_order(scenario)

        gt = get_gt_pred_label(row)
        pred = get_pred_label(row)

        if gt not in order or pred not in order:
            continue

        denom = max(len(order) - 1, 1)
        values.append(abs(order[pred] - order[gt]) / denom)

    if not values:
        return ""

    return sum(values) / len(values)


def format_float(value):
    if value == "":
        return ""
    return f"{float(value):.4f}"


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


with PRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

summary_rows = []

overall_total = len(rows)
overall_correct = sum(get_gt_pred_label(r) == get_pred_label(r) for r in rows)

summary_rows.append({
    "metric_group": "overall",
    "scenario": "all",
    "metric": "severity_exact_accuracy",
    "value": format_float(safe_rate(overall_correct, overall_total)),
    "numerator": overall_correct,
    "denominator": overall_total,
})

summary_rows.append({
    "metric_group": "overall",
    "scenario": "all",
    "metric": "mean_ncse",
    "value": format_float(mean_ncse(rows)),
    "numerator": "",
    "denominator": "",
})

pred_mode_correct = sum(clean(r["pred_mode"]) == clean(r["gt_pred_mode"]) for r in rows)

summary_rows.append({
    "metric_group": "routing",
    "scenario": "all",
    "metric": "gpa_pred_mode_agreement",
    "value": format_float(safe_rate(pred_mode_correct, overall_total)),
    "numerator": pred_mode_correct,
    "denominator": overall_total,
})

analysis_mode_correct = sum(clean(r["analysis_mode"]) == clean(r["scenario"]) for r in rows)

summary_rows.append({
    "metric_group": "routing",
    "scenario": "all",
    "metric": "analysis_mode_route_accuracy",
    "value": format_float(safe_rate(analysis_mode_correct, overall_total)),
    "numerator": analysis_mode_correct,
    "denominator": overall_total,
})

guard_applied = sum(clean(r["final_report_guard_applied"]).lower() == "true" for r in rows)

summary_rows.append({
    "metric_group": "report_guardrail",
    "scenario": "all",
    "metric": "guardrail_application_rate",
    "value": format_float(safe_rate(guard_applied, overall_total)),
    "numerator": guard_applied,
    "denominator": overall_total,
})

by_scenario = defaultdict(list)

for row in rows:
    by_scenario[row["scenario"]].append(row)

for scenario, scenario_rows in by_scenario.items():
    total = len(scenario_rows)
    correct = sum(get_gt_pred_label(r) == get_pred_label(r) for r in scenario_rows)

    summary_rows.append({
        "metric_group": "severity",
        "scenario": scenario,
        "metric": "exact_accuracy",
        "value": format_float(safe_rate(correct, total)),
        "numerator": correct,
        "denominator": total,
    })

    summary_rows.append({
        "metric_group": "severity",
        "scenario": scenario,
        "metric": "mean_ncse",
        "value": format_float(mean_ncse(scenario_rows)),
        "numerator": "",
        "denominator": "",
    })

    route_correct = sum(clean(r["analysis_mode"]) == clean(r["scenario"]) for r in scenario_rows)

    summary_rows.append({
        "metric_group": "routing",
        "scenario": scenario,
        "metric": "analysis_mode_route_accuracy",
        "value": format_float(safe_rate(route_correct, total)),
        "numerator": route_correct,
        "denominator": total,
    })

    guard_count = sum(clean(r["final_report_guard_applied"]).lower() == "true" for r in scenario_rows)

    summary_rows.append({
        "metric_group": "report_guardrail",
        "scenario": scenario,
        "metric": "guardrail_application_rate",
        "value": format_float(safe_rate(guard_count, total)),
        "numerator": guard_count,
        "denominator": total,
    })

with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["metric_group", "scenario", "metric", "value", "numerator", "denominator"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

confusion_rows = []

for scenario, scenario_rows in by_scenario.items():
    counter = Counter()

    for row in scenario_rows:
        gt = get_gt_pred_label(row)
        pred = get_pred_label(row)
        counter[(gt, pred)] += 1

    for (gt, pred), count in sorted(counter.items()):
        confusion_rows.append({
            "scenario": scenario,
            "gt_label": gt,
            "pred_label": pred,
            "count": count,
        })

with CONFUSION_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["scenario", "gt_label", "pred_label", "count"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(confusion_rows)

lines = []
lines.append("HERMES Smoke Evaluation Metrics")
lines.append("=" * 40)
lines.append(f"Total cases: {overall_total}")
lines.append(f"Severity exact accuracy: {overall_correct}/{overall_total} = {safe_rate(overall_correct, overall_total):.4f}")
lines.append(f"Mean NCSE: {format_float(mean_ncse(rows))}")
lines.append(f"GPA pred_mode agreement: {pred_mode_correct}/{overall_total} = {safe_rate(pred_mode_correct, overall_total):.4f}")
lines.append(f"Analysis-mode route accuracy: {analysis_mode_correct}/{overall_total} = {safe_rate(analysis_mode_correct, overall_total):.4f}")
lines.append(f"Guardrail application rate: {guard_applied}/{overall_total} = {safe_rate(guard_applied, overall_total):.4f}")
lines.append("")
lines.append("By scenario")
lines.append("-" * 40)

for scenario, scenario_rows in by_scenario.items():
    total = len(scenario_rows)
    correct = sum(get_gt_pred_label(r) == get_pred_label(r) for r in scenario_rows)
    route_correct = sum(clean(r["analysis_mode"]) == clean(r["scenario"]) for r in scenario_rows)
    guard_count = sum(clean(r["final_report_guard_applied"]).lower() == "true" for r in scenario_rows)

    lines.append(f"{scenario}")
    lines.append(f"  cases: {total}")
    lines.append(f"  exact accuracy: {correct}/{total} = {safe_rate(correct, total):.4f}")
    lines.append(f"  mean NCSE: {format_float(mean_ncse(scenario_rows))}")
    lines.append(f"  route accuracy: {route_correct}/{total} = {safe_rate(route_correct, total):.4f}")
    lines.append(f"  guardrail rate: {guard_count}/{total} = {safe_rate(guard_count, total):.4f}")

SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Metrics summary CSV written to: {SUMMARY_CSV}")
print(f"Confusion matrix CSV written to: {CONFUSION_CSV}")
print(f"Metrics summary TXT written to: {SUMMARY_TXT}")

print("\n" + "\n".join(lines))
