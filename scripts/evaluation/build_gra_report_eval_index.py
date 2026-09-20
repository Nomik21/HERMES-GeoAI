from pathlib import Path
import json
import html

ROOT = Path.cwd() / "evaluation" / "gra_report_evaluation_set"
INDEX_PATH = ROOT / "index.html"

SCENARIOS = [
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
    "object_detection",
]

rows = []

for scenario in SCENARIOS:
    reports_dir = ROOT / scenario / "reports"
    if not reports_dir.exists():
        continue

    for case_dir in sorted([p for p in reports_dir.iterdir() if p.is_dir()]):
        json_files = list(case_dir.glob("*.json"))
        html_files = list(case_dir.glob("*.html"))

        if not json_files or not html_files:
            continue

        data = {}
        try:
            data = json.loads(json_files[0].read_text(encoding="utf-8"))
        except Exception:
            data = {}

        report_html = html_files[0]
        rel_link = report_html.relative_to(ROOT).as_posix()

        case_id = data.get("case_id", case_dir.name)
        hazard = data.get("type_of_disaster", "")
        input_mode = data.get("input_mode", "")
        report_scenario = data.get("report_scenario", scenario)

        damage = data.get("damage_assessment", {}) or {}
        severity = (
            damage.get("severity")
            or damage.get("overall_severity")
            or damage.get("three_level_severity")
            or ""
        )

        executive_summary = data.get("executive_summary", "")
        if len(executive_summary) > 220:
            executive_summary = executive_summary[:220] + "..."

        rows.append({
            "scenario": scenario,
            "case_id": case_id,
            "hazard": hazard,
            "input_mode": input_mode,
            "report_scenario": report_scenario,
            "severity": severity,
            "summary": executive_summary,
            "link": rel_link,
        })

html_rows = []

for i, r in enumerate(rows, start=1):
    html_rows.append(f"""
    <tr>
      <td>{i}</td>
      <td>{html.escape(r["scenario"])}</td>
      <td>{html.escape(str(r["case_id"]))}</td>
      <td>{html.escape(str(r["hazard"]))}</td>
      <td>{html.escape(str(r["input_mode"]))}</td>
      <td>{html.escape(str(r["severity"]))}</td>
      <td>{html.escape(str(r["summary"]))}</td>
      <td><a href="{html.escape(r["link"])}" target="_blank">Open report</a></td>
    </tr>
    """)

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HERMES GRA Report Evaluation Set</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 32px;
    background: #f7f7f7;
    color: #222;
}}
h1 {{
    margin-bottom: 4px;
}}
.subtitle {{
    margin-bottom: 24px;
    color: #555;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px;
    vertical-align: top;
    font-size: 14px;
}}
th {{
    background: #222;
    color: white;
    position: sticky;
    top: 0;
}}
tr:nth-child(even) {{
    background: #f2f2f2;
}}
a {{
    font-weight: bold;
}}
.badge {{
    display: inline-block;
    padding: 4px 8px;
    background: #e9e9e9;
    border-radius: 6px;
    margin-right: 8px;
}}
</style>
</head>
<body>
<h1>HERMES GRA Report Evaluation Set</h1>
<div class="subtitle">
  <span class="badge">Total reports: {len(rows)}</span>
  <span class="badge">Expected: 40</span>
  <span class="badge">Scenarios: 4 × 10</span>
</div>

<table>
<thead>
<tr>
  <th>#</th>
  <th>Scenario</th>
  <th>Case ID</th>
  <th>Hazard</th>
  <th>Input Mode</th>
  <th>Severity</th>
  <th>Executive Summary</th>
  <th>Report</th>
</tr>
</thead>
<tbody>
{''.join(html_rows)}
</tbody>
</table>
</body>
</html>
"""

INDEX_PATH.write_text(page, encoding="utf-8")

print(f"Index created: {INDEX_PATH}")
print(f"Total reports indexed: {len(rows)}")
print("DONE ✅")
