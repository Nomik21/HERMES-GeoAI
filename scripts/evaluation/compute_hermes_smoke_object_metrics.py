from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRED_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_final_predictions.csv"
OBJECT_GT_CSV = PROJECT_ROOT / "evaluation" / "hermes_object_groundtruth.csv"

OBJECT_CASES_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_object_indicator_cases.csv"
OBJECT_METRICS_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_object_indicator_metrics.csv"
OBJECT_METRICS_TXT = PROJECT_ROOT / "evaluation" / "hermes_smoke_object_indicator_metrics.txt"

OBJECT_MAPPING = {
    "Obj_Debris": ["debris_pile", "debris"],
    "Obj_FallenTree": ["fallen_tree"],
    "Obj_Flooded": ["flooded_road", "flooding"],
    "Obj_DamagedBldg": ["damaged_building", "damaged_house"],
    "Obj_DownedLines": ["downed_lines"],
}

OBJECT_COLUMNS = list(OBJECT_MAPPING.keys())


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_objects(value):
    text = clean(value)

    if not text:
        return {}

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def object_positive(objects_detected, keys):
    for key in keys:
        try:
            if float(objects_detected.get(key, 0) or 0) > 0:
                return 1
        except (TypeError, ValueError):
            continue

    return 0


def safe_div(num, den):
    return num / den if den else 0.0


def f4(value):
    return f"{value:.4f}"


with PRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    pred_rows = list(csv.DictReader(f))

with OBJECT_GT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    gt_rows = list(csv.DictReader(f))

gt_lookup = {
    (row["scenario"], row["case_id"]): row
    for row in gt_rows
}

case_rows = []

for pred in pred_rows:
    scenario = pred["scenario"]

    if scenario not in {"cross_view_svi_rsi", "pre_post_svi"}:
        continue

    key = (scenario, pred["case_id"])
    gt = gt_lookup.get(key)

    if gt is None:
        continue

    objects_detected = parse_objects(pred["objects_detected_json"])

    out = {
        "case_id": pred["case_id"],
        "scenario": scenario,
        "gt_severity_label": pred["gt_severity_label"],
        "pred_severity_label": pred["pred_severity_label"],
    }

    for obj_col, pred_keys in OBJECT_MAPPING.items():
        gt_value = int(float(gt.get(obj_col, 0) or 0))
        pred_value = object_positive(objects_detected, pred_keys)

        out[f"gt_{obj_col}"] = gt_value
        out[f"pred_{obj_col}"] = pred_value
        out[f"match_{obj_col}"] = int(gt_value == pred_value)

    out["objects_detected_json"] = pred["objects_detected_json"]

    case_rows.append(out)

case_fieldnames = [
    "case_id",
    "scenario",
    "gt_severity_label",
    "pred_severity_label",
]

for obj_col in OBJECT_COLUMNS:
    case_fieldnames.extend([
        f"gt_{obj_col}",
        f"pred_{obj_col}",
        f"match_{obj_col}",
    ])

case_fieldnames.append("objects_detected_json")

with OBJECT_CASES_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=case_fieldnames)
    writer.writeheader()
    writer.writerows(case_rows)

metric_rows = []

micro_tp = micro_fp = micro_fn = micro_tn = 0

for obj_col in OBJECT_COLUMNS:
    tp = fp = fn = tn = 0

    for row in case_rows:
        gt = int(row[f"gt_{obj_col}"])
        pred = int(row[f"pred_{obj_col}"])

        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        elif gt == 0 and pred == 0:
            tn += 1

    support_pos = tp + fn
    support_neg = tn + fp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    micro_tp += tp
    micro_fp += fp
    micro_fn += fn
    micro_tn += tn

    metric_rows.append({
        "object_indicator": obj_col,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_positive": support_pos,
        "support_negative": support_neg,
        "precision": f4(precision),
        "recall": f4(recall),
        "f1": f4(f1),
        "accuracy": f4(accuracy),
    })

macro_precision = sum(float(r["precision"]) for r in metric_rows) / len(metric_rows)
macro_recall = sum(float(r["recall"]) for r in metric_rows) / len(metric_rows)
macro_f1 = sum(float(r["f1"]) for r in metric_rows) / len(metric_rows)

micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
micro_accuracy = safe_div(micro_tp + micro_tn, micro_tp + micro_tn + micro_fp + micro_fn)

metric_rows.append({
    "object_indicator": "MACRO_AVERAGE",
    "tp": "",
    "fp": "",
    "fn": "",
    "tn": "",
    "support_positive": "",
    "support_negative": "",
    "precision": f4(macro_precision),
    "recall": f4(macro_recall),
    "f1": f4(macro_f1),
    "accuracy": "",
})

metric_rows.append({
    "object_indicator": "MICRO_AVERAGE",
    "tp": micro_tp,
    "fp": micro_fp,
    "fn": micro_fn,
    "tn": micro_tn,
    "support_positive": micro_tp + micro_fn,
    "support_negative": micro_tn + micro_fp,
    "precision": f4(micro_precision),
    "recall": f4(micro_recall),
    "f1": f4(micro_f1),
    "accuracy": f4(micro_accuracy),
})

with OBJECT_METRICS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "object_indicator",
        "tp",
        "fp",
        "fn",
        "tn",
        "support_positive",
        "support_negative",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metric_rows)

lines = []
lines.append("HERMES Smoke Object-Indicator Evaluation")
lines.append("=" * 50)
lines.append(f"Object-labeled smoke cases: {len(case_rows)}")
lines.append("")
lines.append("Per-indicator metrics")
lines.append("-" * 50)

for row in metric_rows:
    lines.append(
        f"{row['object_indicator']}: "
        f"P={row['precision']} | R={row['recall']} | F1={row['f1']} | Acc={row['accuracy']} | "
        f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}"
    )

OBJECT_METRICS_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Object case CSV written to: {OBJECT_CASES_CSV}")
print(f"Object metrics CSV written to: {OBJECT_METRICS_CSV}")
print(f"Object metrics TXT written to: {OBJECT_METRICS_TXT}")

print("\n" + "\n".join(lines))
