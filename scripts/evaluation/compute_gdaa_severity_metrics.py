from pathlib import Path
import argparse
import csv
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.csv"
DEFAULT_METRICS = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_metrics.csv"
DEFAULT_CONFUSION = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_confusion_matrix.csv"
DEFAULT_BY_SCENARIO = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_by_scenario.csv"
DEFAULT_TXT = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_metrics.txt"

SCENARIO_LABELS = {
    "cross_view_svi_rsi": [
        ("0", "Mild"),
        ("1", "Moderate"),
        ("2", "Severe"),
    ],
    "pre_post_svi": [
        ("0", "Mild"),
        ("1", "Moderate"),
        ("2", "Severe"),
    ],
    "wildfire_building": [
        ("0", "0_No_Damage"),
        ("1", "1_Affected_1_9"),
        ("2", "2_Minor_10_25"),
        ("3", "3_Major_26_50"),
        ("4", "4_Destroyed_50plus"),
    ],
}


def resolve(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_latest_success_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    latest = {}
    for row in rows:
        latest[row["case_id"]] = row

    latest_rows = list(latest.values())
    success_rows = [r for r in latest_rows if r.get("status") == "success"]

    return rows, latest_rows, success_rows


def safe_int(value):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metrics(rows):
    valid_rows = []

    for row in rows:
        gt = safe_int(row.get("gt_severity_id"))
        pred = safe_int(row.get("pred_severity_id"))
        scenario = row.get("scenario", "")

        if gt is None or pred is None:
            continue

        if scenario not in SCENARIO_LABELS:
            continue

        row["_gt_id_int"] = gt
        row["_pred_id_int"] = pred
        valid_rows.append(row)

    total = len(valid_rows)
    correct = sum(1 for r in valid_rows if r["_gt_id_int"] == r["_pred_id_int"])
    accuracy = correct / total if total else 0.0

    ncse_values = []
    for r in valid_rows:
        k = len(SCENARIO_LABELS[r["scenario"]])
        ncse_values.append(abs(r["_gt_id_int"] - r["_pred_id_int"]) / (k - 1))

    mean_ncse = sum(ncse_values) / len(ncse_values) if ncse_values else 0.0

    scenario_rows = defaultdict(list)
    for r in valid_rows:
        scenario_rows[r["scenario"]].append(r)

    by_scenario = []

    for scenario, srows in sorted(scenario_rows.items()):
        s_total = len(srows)
        s_correct = sum(1 for r in srows if r["_gt_id_int"] == r["_pred_id_int"])
        k = len(SCENARIO_LABELS[scenario])
        s_ncse = sum(
            abs(r["_gt_id_int"] - r["_pred_id_int"]) / (k - 1)
            for r in srows
        ) / s_total if s_total else 0.0

        by_scenario.append({
            "scenario": scenario,
            "total": s_total,
            "correct": s_correct,
            "accuracy": s_correct / s_total if s_total else 0.0,
            "mean_ncse": s_ncse,
        })

    per_class = []

    for scenario, labels in SCENARIO_LABELS.items():
        srows = scenario_rows.get(scenario, [])

        for label_id_text, label_name in labels:
            label_id = int(label_id_text)

            tp = sum(
                1 for r in srows
                if r["_gt_id_int"] == label_id and r["_pred_id_int"] == label_id
            )
            fp = sum(
                1 for r in srows
                if r["_gt_id_int"] != label_id and r["_pred_id_int"] == label_id
            )
            fn = sum(
                1 for r in srows
                if r["_gt_id_int"] == label_id and r["_pred_id_int"] != label_id
            )
            tn = sum(
                1 for r in srows
                if r["_gt_id_int"] != label_id and r["_pred_id_int"] != label_id
            )

            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            binary_acc = (tp + tn) / len(srows) if srows else 0.0
            support = sum(1 for r in srows if r["_gt_id_int"] == label_id)

            per_class.append({
                "scenario": scenario,
                "label_id": label_id,
                "label": label_name,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "binary_accuracy": binary_acc,
            })

    confusion_rows = []

    for scenario, labels in SCENARIO_LABELS.items():
        srows = scenario_rows.get(scenario, [])
        label_ids = [int(x[0]) for x in labels]

        for gt_id in label_ids:
            out = {
                "scenario": scenario,
                "gt_id": gt_id,
                "gt_label": dict((int(i), name) for i, name in labels)[gt_id],
            }

            for pred_id in label_ids:
                out[f"pred_{pred_id}"] = sum(
                    1 for r in srows
                    if r["_gt_id_int"] == gt_id and r["_pred_id_int"] == pred_id
                )

            confusion_rows.append(out)

    return {
        "valid_rows": valid_rows,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "mean_ncse": mean_ncse,
        "by_scenario": by_scenario,
        "per_class": per_class,
        "confusion_rows": confusion_rows,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--confusion", default=str(DEFAULT_CONFUSION))
    parser.add_argument("--by-scenario", default=str(DEFAULT_BY_SCENARIO))
    parser.add_argument("--txt", default=str(DEFAULT_TXT))
    args = parser.parse_args()

    input_path = resolve(args.input)
    metrics_path = resolve(args.metrics)
    confusion_path = resolve(args.confusion)
    by_scenario_path = resolve(args.by_scenario)
    txt_path = resolve(args.txt)

    all_rows, latest_rows, success_rows = read_latest_success_rows(input_path)
    result = compute_metrics(success_rows)

    per_class = result["per_class"]
    by_scenario = result["by_scenario"]
    confusion_rows = result["confusion_rows"]

    write_csv(
        metrics_path,
        per_class,
        [
            "scenario",
            "label_id",
            "label",
            "support",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "binary_accuracy",
        ],
    )

    confusion_fieldnames = [
        "scenario",
        "gt_id",
        "gt_label",
        "pred_0",
        "pred_1",
        "pred_2",
        "pred_3",
        "pred_4",
    ]

    # Some scenarios have only pred_0..pred_2; fill missing columns.
    for row in confusion_rows:
        for col in confusion_fieldnames:
            row.setdefault(col, "")

    write_csv(confusion_path, confusion_rows, confusion_fieldnames)

    write_csv(
        by_scenario_path,
        by_scenario,
        ["scenario", "total", "correct", "accuracy", "mean_ncse"],
    )

    lines = []
    lines.append("HERMES GDAA Severity Evaluation")
    lines.append("=" * 75)
    lines.append(f"Input CSV: {input_path}")
    lines.append(f"Total CSV rows: {len(all_rows)}")
    lines.append(f"Unique cases: {len(latest_rows)}")
    lines.append(f"Successful cases: {len(success_rows)}")
    lines.append(f"Valid evaluated cases: {result['total']}")
    lines.append("")
    lines.append(f"Severity exact accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")
    lines.append(f"Mean NCSE: {result['mean_ncse']:.4f}")
    lines.append("")
    lines.append("By scenario")
    lines.append("-" * 75)

    for item in by_scenario:
        lines.append(
            f"{item['scenario']}: "
            f"{item['correct']}/{item['total']} = {item['accuracy']:.4f} | "
            f"mean NCSE = {item['mean_ncse']:.4f}"
        )

    lines.append("")
    lines.append("Per-class metrics")
    lines.append("-" * 75)

    for item in per_class:
        if item["support"] == 0:
            continue

        lines.append(
            f"{item['scenario']} | {item['label']}: "
            f"P={item['precision']:.4f} | "
            f"R={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | "
            f"support={item['support']} | "
            f"TP={item['tp']} FP={item['fp']} FN={item['fn']}"
        )

    lines.append("")
    lines.append("Output files")
    lines.append("-" * 75)
    lines.append(f"Metrics CSV: {metrics_path}")
    lines.append(f"Confusion matrix CSV: {confusion_path}")
    lines.append(f"By-scenario CSV: {by_scenario_path}")
    lines.append(f"Summary TXT: {txt_path}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
