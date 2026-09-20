from pathlib import Path
import argparse
import csv
import json
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GT = PROJECT_ROOT / "evaluation" / "hermes_object_groundtruth.csv"
DEFAULT_JSONL = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_full_results.jsonl"

DEFAULT_PREDICTIONS = PROJECT_ROOT / "evaluation" / "hermes_gdaa_object_indicator_full_predictions.csv"
DEFAULT_METRICS = PROJECT_ROOT / "evaluation" / "hermes_gdaa_object_indicator_full_metrics.csv"
DEFAULT_BY_SCENARIO = PROJECT_ROOT / "evaluation" / "hermes_gdaa_object_indicator_full_by_scenario.csv"
DEFAULT_TXT = PROJECT_ROOT / "evaluation" / "hermes_gdaa_object_indicator_full_summary.txt"

OBJECT_MAPPING = {
    "Obj_Debris": ["debris_pile"],
    "Obj_FallenTree": ["fallen_tree"],
    "Obj_Flooded": ["flooded_road", "flooding"],
    "Obj_DamagedBldg": ["damaged_building", "roof_damage", "structural_damage"],
    "Obj_DownedLines": ["downed_lines"],
}

OBJECT_COLUMNS = list(OBJECT_MAPPING.keys())


def resolve(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_binary(value):
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0

    text = str(value or "").strip().lower()

    if text in {"1", "true", "yes", "present", "visible"}:
        return 1

    if text in {"0", "false", "no", "absent", "not visible", ""}:
        return 0

    return 0


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_latest(path):
    latest = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            case_id = str(payload.get("case_id", "")).strip()

            if case_id:
                latest[case_id] = payload

    return latest


def extract_objects(result):
    if not isinstance(result, dict):
        return {}

    objects = (
        result.get("objects_detected")
        or result.get("Objects_Detected")
        or result.get("objects")
        or result.get("Objects")
        or {}
    )

    if not isinstance(objects, dict):
        return {}

    return objects


def extract_pred_for_gt_column(objects, gt_col):
    pred_keys = OBJECT_MAPPING[gt_col]

    for key in pred_keys:
        if to_binary(objects.get(key, 0)) == 1:
            return 1

    return 0


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metric_rows(pred_rows, scenario_filter=None):
    rows = pred_rows

    if scenario_filter is not None:
        rows = [r for r in rows if r["scenario"] == scenario_filter]

    metric_rows = []

    total_tp = total_fp = total_fn = total_tn = 0

    for col in OBJECT_COLUMNS:
        tp = fp = fn = tn = 0

        gt_key = f"gt_{col}"
        pred_key = f"pred_{col}"

        for row in rows:
            gt = int(row[gt_key])
            pred = int(row[pred_key])

            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 1 and pred == 0:
                fn += 1
            elif gt == 0 and pred == 0:
                tn += 1

        precision, recall, f1 = prf(tp, fp, fn)
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
        support = tp + fn

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        metric_rows.append({
            "scenario": scenario_filter or "ALL",
            "object_indicator": col,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "binary_accuracy": acc,
        })

    macro_precision = sum(r["precision"] for r in metric_rows) / len(metric_rows) if metric_rows else 0.0
    macro_recall = sum(r["recall"] for r in metric_rows) / len(metric_rows) if metric_rows else 0.0
    macro_f1 = sum(r["f1"] for r in metric_rows) / len(metric_rows) if metric_rows else 0.0

    micro_precision, micro_recall, micro_f1 = prf(total_tp, total_fp, total_fn)
    micro_acc = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) else 0.0

    aggregate = {
        "scenario": scenario_filter or "ALL",
        "cases": len(rows),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_accuracy": micro_acc,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }

    return metric_rows, aggregate


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--by-scenario", default=str(DEFAULT_BY_SCENARIO))
    parser.add_argument("--txt", default=str(DEFAULT_TXT))
    args = parser.parse_args()

    gt_path = resolve(args.gt)
    jsonl_path = resolve(args.jsonl)
    predictions_path = resolve(args.predictions)
    metrics_path = resolve(args.metrics)
    by_scenario_path = resolve(args.by_scenario)
    txt_path = resolve(args.txt)

    gt_rows = read_csv(gt_path)
    latest_payloads = read_jsonl_latest(jsonl_path)

    pred_rows = []
    missing_results = []

    for gt_row in gt_rows:
        case_id = str(gt_row["case_id"]).strip()
        scenario = str(gt_row["scenario"]).strip()

        payload = latest_payloads.get(case_id)

        if not payload:
            missing_results.append(case_id)
            continue

        result = payload.get("result", {})
        objects = extract_objects(result)

        out = {
            "case_id": case_id,
            "scenario": scenario,
            "dataset": gt_row.get("dataset", ""),
            "status": payload.get("status", "success"),
        }

        for col in OBJECT_COLUMNS:
            out[f"gt_{col}"] = to_binary(gt_row.get(col, 0))
            out[f"pred_{col}"] = extract_pred_for_gt_column(objects, col)

        pred_rows.append(out)

    scenarios = sorted(set(r["scenario"] for r in pred_rows))

    all_metric_rows = []
    by_scenario_rows = []

    metric_rows, aggregate = compute_metric_rows(pred_rows, scenario_filter=None)
    all_metric_rows.extend(metric_rows)
    by_scenario_rows.append(aggregate)

    for scenario in scenarios:
        metric_rows, aggregate = compute_metric_rows(pred_rows, scenario_filter=scenario)
        all_metric_rows.extend(metric_rows)
        by_scenario_rows.append(aggregate)

    prediction_fields = [
        "case_id",
        "scenario",
        "dataset",
        "status",
    ]

    for col in OBJECT_COLUMNS:
        prediction_fields.extend([f"gt_{col}", f"pred_{col}"])

    metric_fields = [
        "scenario",
        "object_indicator",
        "support",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "binary_accuracy",
    ]

    by_scenario_fields = [
        "scenario",
        "cases",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    write_csv(predictions_path, pred_rows, prediction_fields)
    write_csv(metrics_path, all_metric_rows, metric_fields)
    write_csv(by_scenario_path, by_scenario_rows, by_scenario_fields)

    lines = []
    lines.append("HERMES GDAA Object-Indicator Full Evaluation")
    lines.append("=" * 80)
    lines.append(f"Object GT CSV: {gt_path}")
    lines.append(f"GDAA JSONL: {jsonl_path}")
    lines.append(f"GT cases: {len(gt_rows)}")
    lines.append(f"Evaluated cases: {len(pred_rows)}")
    lines.append(f"Missing GDAA results: {len(missing_results)}")
    lines.append("")

    lines.append("Overall")
    lines.append("-" * 80)
    overall = by_scenario_rows[0]
    lines.append(
        f"Cases={overall['cases']} | "
        f"Macro F1={overall['macro_f1']:.4f} | "
        f"Micro F1={overall['micro_f1']:.4f} | "
        f"Micro Accuracy={overall['micro_accuracy']:.4f}"
    )
    lines.append(
        f"Micro Precision={overall['micro_precision']:.4f} | "
        f"Micro Recall={overall['micro_recall']:.4f}"
    )
    lines.append("")

    lines.append("By scenario")
    lines.append("-" * 80)
    for item in by_scenario_rows[1:]:
        lines.append(
            f"{item['scenario']}: "
            f"cases={item['cases']} | "
            f"Macro F1={item['macro_f1']:.4f} | "
            f"Micro F1={item['micro_f1']:.4f} | "
            f"Micro Accuracy={item['micro_accuracy']:.4f}"
        )

    lines.append("")
    lines.append("Per-object indicators, overall")
    lines.append("-" * 80)
    for row in all_metric_rows:
        if row["scenario"] != "ALL":
            continue

        lines.append(
            f"{row['object_indicator']}: "
            f"P={row['precision']:.4f} | "
            f"R={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"Acc={row['binary_accuracy']:.4f} | "
            f"support={row['support']} | "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}"
        )

    lines.append("")
    lines.append("Output files")
    lines.append("-" * 80)
    lines.append(f"Predictions CSV: {predictions_path}")
    lines.append(f"Metrics CSV: {metrics_path}")
    lines.append(f"By-scenario CSV: {by_scenario_path}")
    lines.append(f"Summary TXT: {txt_path}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
