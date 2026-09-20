from pathlib import Path
import csv
from collections import OrderedDict, Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_results.csv"

METRICS_CSV = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_metrics.csv"
CONFUSION_CSV = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_confusion_matrix.csv"
BY_SCENARIO_CSV = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_by_scenario.csv"
SUMMARY_TXT = PROJECT_ROOT / "evaluation" / "hermes_gpa_mode_full_neutral_metrics.txt"

MODES = [
    "post_sv_single",
    "post_rs_single",
    "post_paired_sv_rs",
    "bitemporal_sv",
    "bitemporal_rs",
]


def norm(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def safe_div(num, den):
    return num / den if den else 0.0


def f4(value):
    return f"{float(value):.4f}"


with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Deduplicate by case_id, keeping latest row.
latest_by_case = OrderedDict()
for row in rows:
    latest_by_case[row["case_id"]] = row

unique_rows = list(latest_by_case.values())
success_rows = [row for row in unique_rows if row.get("status") == "success"]
failed_rows = [row for row in unique_rows if row.get("status") != "success"]

invalid_rows = []
for row in success_rows:
    gt = norm(row.get("gt_pred_mode"))
    pred = norm(row.get("pred_mode"))
    if gt not in MODES or pred not in MODES:
        invalid_rows.append(row)

if invalid_rows:
    print("Invalid mode labels found:")
    for row in invalid_rows[:50]:
        print(
            f"  case_id={row.get('case_id')} | "
            f"gt={row.get('gt_pred_mode')} | pred={row.get('pred_mode')} | "
            f"status={row.get('status')}"
        )
    raise SystemExit("Mode schema validation failed ❌")

total = len(success_rows)
mode_correct = sum(
    norm(row.get("gt_pred_mode")) == norm(row.get("pred_mode"))
    for row in success_rows
)

confusion = {
    gt: {pred: 0 for pred in MODES}
    for gt in MODES
}

for row in success_rows:
    gt = norm(row.get("gt_pred_mode"))
    pred = norm(row.get("pred_mode"))
    confusion[gt][pred] += 1

metric_rows = []

for mode in MODES:
    tp = confusion[mode][mode]
    fp = sum(confusion[gt][mode] for gt in MODES if gt != mode)
    fn = sum(confusion[mode][pred] for pred in MODES if pred != mode)
    tn = total - tp - fp - fn
    support = sum(confusion[mode].values())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    binary_accuracy = safe_div(tp + tn, total)

    metric_rows.append({
        "mode": mode,
        "precision": f4(precision),
        "recall": f4(recall),
        "f1": f4(f1),
        "binary_accuracy": f4(binary_accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
    })

macro_precision = safe_div(sum(float(row["precision"]) for row in metric_rows), len(metric_rows))
macro_recall = safe_div(sum(float(row["recall"]) for row in metric_rows), len(metric_rows))
macro_f1 = safe_div(sum(float(row["f1"]) for row in metric_rows), len(metric_rows))

weighted_precision = safe_div(
    sum(float(row["precision"]) * int(row["support"]) for row in metric_rows),
    total,
)
weighted_recall = safe_div(
    sum(float(row["recall"]) * int(row["support"]) for row in metric_rows),
    total,
)
weighted_f1 = safe_div(
    sum(float(row["f1"]) * int(row["support"]) for row in metric_rows),
    total,
)

overall_accuracy = safe_div(mode_correct, total)

aggregate_rows = [
    {
        "mode": "MACRO_AVERAGE",
        "precision": f4(macro_precision),
        "recall": f4(macro_recall),
        "f1": f4(macro_f1),
        "binary_accuracy": "",
        "tp": "",
        "fp": "",
        "fn": "",
        "tn": "",
        "support": total,
    },
    {
        "mode": "WEIGHTED_AVERAGE",
        "precision": f4(weighted_precision),
        "recall": f4(weighted_recall),
        "f1": f4(weighted_f1),
        "binary_accuracy": "",
        "tp": "",
        "fp": "",
        "fn": "",
        "tn": "",
        "support": total,
    },
    {
        "mode": "MICRO_AVERAGE",
        "precision": f4(overall_accuracy),
        "recall": f4(overall_accuracy),
        "f1": f4(overall_accuracy),
        "binary_accuracy": "",
        "tp": mode_correct,
        "fp": total - mode_correct,
        "fn": total - mode_correct,
        "tn": "",
        "support": total,
    },
    {
        "mode": "OVERALL_MODE_ACCURACY",
        "precision": "",
        "recall": "",
        "f1": "",
        "binary_accuracy": f4(overall_accuracy),
        "tp": mode_correct,
        "fp": total - mode_correct,
        "fn": "",
        "tn": "",
        "support": total,
    },
]

with METRICS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "mode",
        "precision",
        "recall",
        "f1",
        "binary_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "support",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metric_rows + aggregate_rows)

with CONFUSION_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["gt_mode"] + MODES
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for gt in MODES:
        out = {"gt_mode": gt}
        out.update(confusion[gt])
        writer.writerow(out)

scenario_groups = defaultdict(list)
for row in success_rows:
    scenario_groups[row["scenario"]].append(row)

scenario_rows = []

for scenario, group in sorted(scenario_groups.items()):
    correct = sum(norm(row["gt_pred_mode"]) == norm(row["pred_mode"]) for row in group)
    scenario_rows.append({
        "scenario": scenario,
        "cases": len(group),
        "correct": correct,
        "accuracy": f4(safe_div(correct, len(group))),
        "gt_modes": ";".join(sorted(set(norm(row["gt_pred_mode"]) for row in group))),
        "pred_modes": ";".join(sorted(set(norm(row["pred_mode"]) for row in group))),
    })

with BY_SCENARIO_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["scenario", "cases", "correct", "accuracy", "gt_modes", "pred_modes"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(scenario_rows)

gt_counts = Counter(norm(row["gt_pred_mode"]) for row in success_rows)
pred_counts = Counter(norm(row["pred_mode"]) for row in success_rows)

lines = []
lines.append("HERMES GPA Input-Mode Full Evaluation — Neutral Inputs")
lines.append("=" * 75)
lines.append(f"Input CSV: {INPUT_CSV.relative_to(PROJECT_ROOT)}")
lines.append(f"Total CSV rows: {len(rows)}")
lines.append(f"Unique cases: {len(unique_rows)}")
lines.append(f"Successful cases: {total}")
lines.append(f"Failed/non-success cases: {len(failed_rows)}")
lines.append("")
lines.append(f"Mode accuracy: {mode_correct}/{total} = {f4(overall_accuracy)}")
lines.append("")
lines.append("Ground-truth mode distribution")
lines.append("-" * 75)
for mode in MODES:
    lines.append(f"{mode}: {gt_counts[mode]}")
lines.append("")
lines.append("Prediction mode distribution")
lines.append("-" * 75)
for mode in MODES:
    lines.append(f"{mode}: {pred_counts[mode]}")
lines.append("")
lines.append("Per-mode metrics")
lines.append("-" * 75)
for row in metric_rows:
    lines.append(
        f"{row['mode']}: "
        f"P={row['precision']} | "
        f"R={row['recall']} | "
        f"F1={row['f1']} | "
        f"support={row['support']} | "
        f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}"
    )
lines.append("")
lines.append(f"Macro precision: {f4(macro_precision)}")
lines.append(f"Macro recall / balanced accuracy: {f4(macro_recall)}")
lines.append(f"Macro F1: {f4(macro_f1)}")
lines.append("")
lines.append(f"Weighted precision: {f4(weighted_precision)}")
lines.append(f"Weighted recall: {f4(weighted_recall)}")
lines.append(f"Weighted F1: {f4(weighted_f1)}")
lines.append("")
lines.append("By scenario")
lines.append("-" * 75)
for row in scenario_rows:
    lines.append(
        f"{row['scenario']}: "
        f"{row['correct']}/{row['cases']} = {row['accuracy']} | "
        f"gt_modes={row['gt_modes']} | pred_modes={row['pred_modes']}"
    )
lines.append("")
lines.append("Confusion matrix")
lines.append("-" * 75)
lines.append("rows = ground truth, columns = prediction")
lines.append(",".join(["gt\\pred"] + MODES))
for gt in MODES:
    lines.append(",".join([gt] + [str(confusion[gt][pred]) for pred in MODES]))

SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Metrics CSV written to: {METRICS_CSV}")
print(f"Confusion matrix CSV written to: {CONFUSION_CSV}")
print(f"By-scenario CSV written to: {BY_SCENARIO_CSV}")
print(f"Summary TXT written to: {SUMMARY_TXT}")
print()
print("\n".join(lines))
