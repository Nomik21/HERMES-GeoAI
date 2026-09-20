from pathlib import Path
import csv
from collections import OrderedDict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "evaluation" / "hermes_gpa_hazard_full_neutral_results_closedset.csv"

METRICS_CSV = PROJECT_ROOT / "evaluation" / "hermes_gpa_hazard_full_neutral_metrics_closedset.csv"
CONFUSION_CSV = PROJECT_ROOT / "evaluation" / "hermes_gpa_hazard_full_neutral_confusion_matrix_closedset.csv"
SUMMARY_TXT = PROJECT_ROOT / "evaluation" / "hermes_gpa_hazard_full_neutral_metrics_closedset.txt"

LABELS = [
    "drought",
    "earthquake",
    "flood",
    "hurricane",
    "ice_storm",
    "wildfire",
]


def norm(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def safe_div(num, den):
    return num / den if den else 0.0


def f4(value):
    return f"{float(value):.4f}"


with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Deduplicate by case_id, keeping the latest row.
latest_by_case = OrderedDict()
for row in rows:
    latest_by_case[row["case_id"]] = row

unique_rows = list(latest_by_case.values())
success_rows = [row for row in unique_rows if row.get("status") == "success"]
failed_rows = [row for row in unique_rows if row.get("status") != "success"]

invalid_label_rows = []
for row in success_rows:
    gt = norm(row.get("gt_hazard_type"))
    pred = norm(row.get("pred_hazard_type"))

    if gt not in LABELS or pred not in LABELS:
        invalid_label_rows.append(row)

if invalid_label_rows:
    print("Invalid closed-set labels found:")
    for row in invalid_label_rows[:50]:
        print(
            f"  case_id={row.get('case_id')} | "
            f"gt={row.get('gt_hazard_type')} | pred={row.get('pred_hazard_type')}"
        )
    raise SystemExit("Closed-set schema validation failed ❌")

total = len(success_rows)

hazard_correct = sum(
    norm(row.get("gt_hazard_type")) == norm(row.get("pred_hazard_type"))
    for row in success_rows
)

mode_correct = sum(
    norm(row.get("gt_pred_mode")) == norm(row.get("pred_mode"))
    for row in success_rows
)

confusion = {
    gt: {pred: 0 for pred in LABELS}
    for gt in LABELS
}

for row in success_rows:
    gt = norm(row.get("gt_hazard_type"))
    pred = norm(row.get("pred_hazard_type"))
    confusion[gt][pred] += 1

metric_rows = []

for label in LABELS:
    tp = confusion[label][label]
    fp = sum(confusion[gt][label] for gt in LABELS if gt != label)
    fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
    tn = total - tp - fp - fn
    support = sum(confusion[label].values())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    binary_accuracy = safe_div(tp + tn, total)

    metric_rows.append({
        "label": label,
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

overall_accuracy = safe_div(hazard_correct, total)
mode_accuracy = safe_div(mode_correct, total)

# In single-label multi-class classification, micro precision/recall/F1 collapse to overall accuracy.
micro_precision = overall_accuracy
micro_recall = overall_accuracy
micro_f1 = overall_accuracy

aggregate_rows = [
    {
        "label": "MACRO_AVERAGE",
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
        "label": "WEIGHTED_AVERAGE",
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
        "label": "MICRO_AVERAGE",
        "precision": f4(micro_precision),
        "recall": f4(micro_recall),
        "f1": f4(micro_f1),
        "binary_accuracy": "",
        "tp": hazard_correct,
        "fp": total - hazard_correct,
        "fn": total - hazard_correct,
        "tn": "",
        "support": total,
    },
    {
        "label": "OVERALL_HAZARD_ACCURACY",
        "precision": "",
        "recall": "",
        "f1": "",
        "binary_accuracy": f4(overall_accuracy),
        "tp": hazard_correct,
        "fp": total - hazard_correct,
        "fn": "",
        "tn": "",
        "support": total,
    },
    {
        "label": "MODE_ACCURACY",
        "precision": "",
        "recall": "",
        "f1": "",
        "binary_accuracy": f4(mode_accuracy),
        "tp": mode_correct,
        "fp": total - mode_correct,
        "fn": "",
        "tn": "",
        "support": total,
    },
]

all_metric_rows = metric_rows + aggregate_rows

with METRICS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "label",
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
    writer.writerows(all_metric_rows)

with CONFUSION_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["gt_label"] + LABELS
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for gt in LABELS:
        row = {"gt_label": gt}
        row.update(confusion[gt])
        writer.writerow(row)

label_counts = Counter(norm(row.get("gt_hazard_type")) for row in success_rows)
pred_counts = Counter(norm(row.get("pred_hazard_type")) for row in success_rows)

lines = []
lines.append("HERMES GPA Hazard Full Evaluation — Neutral Closed-Set")
lines.append("=" * 75)
lines.append(f"Input CSV: {INPUT_CSV.relative_to(PROJECT_ROOT)}")
lines.append(f"Total CSV rows: {len(rows)}")
lines.append(f"Unique cases: {len(unique_rows)}")
lines.append(f"Successful cases: {total}")
lines.append(f"Failed/non-success cases: {len(failed_rows)}")
lines.append("")
lines.append("Ground-truth distribution")
lines.append("-" * 75)
for label in LABELS:
    lines.append(f"{label}: {label_counts[label]}")
lines.append("")
lines.append("Prediction distribution")
lines.append("-" * 75)
for label in LABELS:
    lines.append(f"{label}: {pred_counts[label]}")
lines.append("")
lines.append(f"Hazard accuracy: {hazard_correct}/{total} = {f4(overall_accuracy)}")
lines.append(f"Mode accuracy: {mode_correct}/{total} = {f4(mode_accuracy)}")
lines.append("")
lines.append("Per-class metrics")
lines.append("-" * 75)

for row in metric_rows:
    lines.append(
        f"{row['label']}: "
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
lines.append(f"Micro precision: {f4(micro_precision)}")
lines.append(f"Micro recall: {f4(micro_recall)}")
lines.append(f"Micro F1: {f4(micro_f1)}")
lines.append("")
lines.append("Confusion matrix")
lines.append("-" * 75)
lines.append("rows = ground truth, columns = prediction")
lines.append(",".join(["gt\\pred"] + LABELS))

for gt in LABELS:
    lines.append(",".join([gt] + [str(confusion[gt][pred]) for pred in LABELS]))

SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Metrics CSV written to: {METRICS_CSV}")
print(f"Confusion matrix CSV written to: {CONFUSION_CSV}")
print(f"Summary TXT written to: {SUMMARY_TXT}")
print()
print("\n".join(lines))
