from pathlib import Path
import csv
from collections import Counter, defaultdict
from statistics import mean

PROJECT_ROOT = Path.cwd()
IN_CSV = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set" / "strict_operational_evaluation" / "gra_strict_operational_scores.csv"
OUT_DIR = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set" / "strict_operational_evaluation"

SUMMARY_TXT = OUT_DIR / "gra_strict_operational_summary.txt"
BY_SCENARIO_CSV = OUT_DIR / "gra_strict_operational_by_scenario.csv"
VERDICT_CSV = OUT_DIR / "gra_strict_operational_verdict_counts.csv"
LOWEST_CSV = OUT_DIR / "gra_strict_operational_lowest_reports.csv"

DIMENSIONS = [
    "operational_specificity",
    "evidence_action_alignment",
    "uncertainty_risk_handling",
    "decision_support_value",
]


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def latest_by_report_path(rows):
    latest = {}
    for row in rows:
        key = row.get("report_json_path") or f"{row.get('scenario')}::{row.get('case_id')}"
        latest[key] = row
    return list(latest.values())


def mean_or_blank(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def fmt(value):
    return "" if value is None else f"{value:.4f}"


def main():
    rows_all = read_rows(IN_CSV)
    rows_latest = latest_by_report_path(rows_all)
    rows_success = [r for r in rows_latest if r.get("status") == "success"]

    overall_scores = {dim: [] for dim in DIMENSIONS}
    overall_means = []

    for row in rows_success:
        for dim in DIMENSIONS:
            overall_scores[dim].append(to_float(row.get(dim)))
        overall_means.append(to_float(row.get("overall_mean")))

    verdict_counts = Counter(row.get("overall_verdict", "") for row in rows_success)

    lines = []
    lines.append("HERMES GRA Strict Operational Quality Evaluation Summary")
    lines.append("=" * 80)
    lines.append(f"Raw CSV rows: {len(rows_all)}")
    lines.append(f"Latest unique reports: {len(rows_latest)}")
    lines.append(f"Successful evaluated reports: {len(rows_success)}")
    lines.append("")
    lines.append("Overall scores")
    lines.append("-" * 80)

    for dim in DIMENSIONS:
        lines.append(f"{dim}: {fmt(mean_or_blank(overall_scores[dim]))}")

    lines.append(f"overall_mean: {fmt(mean_or_blank(overall_means))}")
    lines.append("")
    lines.append("Verdict counts")
    lines.append("-" * 80)

    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"{verdict}: {count}")

    grouped = defaultdict(list)
    for row in rows_success:
        grouped[row.get("scenario", "unknown")].append(row)

    by_scenario_rows = []

    lines.append("")
    lines.append("By scenario")
    lines.append("-" * 80)

    for scenario in sorted(grouped):
        group = grouped[scenario]
        scenario_verdicts = Counter(row.get("overall_verdict", "") for row in group)

        scenario_row = {
            "scenario": scenario,
            "n": len(group),
            "operational_specificity_mean": fmt(mean_or_blank([to_float(r.get("operational_specificity")) for r in group])),
            "evidence_action_alignment_mean": fmt(mean_or_blank([to_float(r.get("evidence_action_alignment")) for r in group])),
            "uncertainty_risk_handling_mean": fmt(mean_or_blank([to_float(r.get("uncertainty_risk_handling")) for r in group])),
            "decision_support_value_mean": fmt(mean_or_blank([to_float(r.get("decision_support_value")) for r in group])),
            "overall_mean": fmt(mean_or_blank([to_float(r.get("overall_mean")) for r in group])),
            "pass": scenario_verdicts.get("pass", 0),
            "borderline": scenario_verdicts.get("borderline", 0),
            "fail": scenario_verdicts.get("fail", 0),
        }

        by_scenario_rows.append(scenario_row)

        lines.append(
            f"{scenario}: n={scenario_row['n']} | "
            f"mean={scenario_row['overall_mean']} | "
            f"OS={scenario_row['operational_specificity_mean']} | "
            f"EAA={scenario_row['evidence_action_alignment_mean']} | "
            f"URH={scenario_row['uncertainty_risk_handling_mean']} | "
            f"DSV={scenario_row['decision_support_value_mean']} | "
            f"pass={scenario_row['pass']} | "
            f"borderline={scenario_row['borderline']} | "
            f"fail={scenario_row['fail']}"
        )

    lowest = sorted(
        rows_success,
        key=lambda r: to_float(r.get("overall_mean")) if to_float(r.get("overall_mean")) is not None else 999,
    )[:12]

    lines.append("")
    lines.append("Lowest scoring reports")
    lines.append("-" * 80)

    for row in lowest:
        lines.append(
            f"{row.get('scenario')} | {row.get('case_id')} | "
            f"mean={row.get('overall_mean')} | verdict={row.get('overall_verdict')} | "
            f"OS={row.get('operational_specificity')} "
            f"EAA={row.get('evidence_action_alignment')} "
            f"URH={row.get('uncertainty_risk_handling')} "
            f"DSV={row.get('decision_support_value')}"
        )

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

    with BY_SCENARIO_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "scenario", "n",
            "operational_specificity_mean",
            "evidence_action_alignment_mean",
            "uncertainty_risk_handling_mean",
            "decision_support_value_mean",
            "overall_mean",
            "pass", "borderline", "fail",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_scenario_rows)

    with VERDICT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["verdict", "count"])
        writer.writeheader()
        for verdict, count in sorted(verdict_counts.items()):
            writer.writerow({"verdict": verdict, "count": count})

    with LOWEST_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "scenario", "case_id", "overall_mean", "overall_verdict",
            "operational_specificity", "evidence_action_alignment",
            "uncertainty_risk_handling", "decision_support_value",
            "major_issues", "recommended_improvements",
            "report_json_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in lowest:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print("\n".join(lines))
    print("")
    print("Files written:")
    print(SUMMARY_TXT)
    print(BY_SCENARIO_CSV)
    print(VERDICT_CSV)
    print(LOWEST_CSV)
    print("DONE ✅")


if __name__ == "__main__":
    main()
