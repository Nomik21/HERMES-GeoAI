from pathlib import Path
import json
from collections import defaultdict

PROJECT_ROOT = Path.cwd()
ROOT = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set"

SCENARIOS = [
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
    "object_detection",
]

print("HERMES GRA Report Evaluation Set Audit")
print("=" * 80)
print(f"Root: {ROOT}")
print("")

total_ok = 0
total_bad = 0

for scenario in SCENARIOS:
    reports_dir = ROOT / scenario / "reports"
    case_dirs = sorted([p for p in reports_dir.iterdir() if p.is_dir()]) if reports_dir.exists() else []

    print("-" * 80)
    print(f"{scenario}")
    print("-" * 80)
    print(f"Report folders: {len(case_dirs)}")

    scenario_ok = 0
    scenario_bad = 0

    for case_dir in case_dirs:
        json_files = list(case_dir.glob("*.json"))
        md_files = list(case_dir.glob("*.md"))
        html_files = list(case_dir.glob("*.html"))

        has_json = len(json_files) >= 1
        has_md = len(md_files) >= 1
        has_html = len(html_files) >= 1

        json_valid = False
        has_final_report = False
        report_scenario = ""

        if has_json:
            try:
                data = json.loads(json_files[0].read_text(encoding="utf-8"))
                json_valid = True
                has_final_report = bool(str(data.get("final_report", "")).strip())
                report_scenario = str(data.get("report_scenario", ""))
            except Exception:
                json_valid = False

        ok = has_json and has_md and has_html and json_valid and has_final_report

        if ok:
            scenario_ok += 1
            total_ok += 1
        else:
            scenario_bad += 1
            total_bad += 1
            print(f"  ISSUE: {case_dir.name}")
            print(f"    json={has_json}, md={has_md}, html={has_html}, json_valid={json_valid}, final_report={has_final_report}, report_scenario={report_scenario}")

    print(f"OK reports: {scenario_ok}")
    print(f"Issues: {scenario_bad}")
    print("")

print("=" * 80)
print(f"TOTAL OK reports: {total_ok}")
print(f"TOTAL issues: {total_bad}")

if total_ok == 40 and total_bad == 0:
    print("FINAL STATUS: 4 × 10 reports are ready ✅")
else:
    print("FINAL STATUS: needs attention ⚠️")
