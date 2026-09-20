from pathlib import Path
import csv
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CSV = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest_smoke.csv"
OUTPUT_TXT = PROJECT_ROOT / "evaluation" / "hermes_smoke_commands_preview.txt"

SUPPORTED_SCENARIOS = {
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
}

CLI_SCENARIO_ARG = {
    "cross_view_svi_rsi": "auto",
    "pre_post_svi": "auto",
    "wildfire_building": "auto",
}

commands = []
skipped = []

with SMOKE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        scenario = row["scenario"]

        if scenario not in SUPPORTED_SCENARIOS:
            skipped.append(row)
            continue

        image_paths = row["image_paths"]
        source_types = row["source_types"]
        hazard_type = row["gt_hazard_type"]
        cli_scenario = CLI_SCENARIO_ARG[scenario]
        pred_mode = row["gt_pred_mode"]

        cmd = (
            f'.\\.venv\\Scripts\\python.exe -m hermes.main '
            f'"{image_paths}" '
            f'"{source_types}" '
            f'{hazard_type} '
            f'{cli_scenario} '
            f'{pred_mode} '
            f'true'
        )

        commands.append({
            "case_id": row["case_id"],
            "scenario": scenario,
            "gt_hazard_type": row["gt_hazard_type"],
            "gt_severity_label": row["gt_severity_label"],
            "command": cmd,
        })

with OUTPUT_TXT.open("w", encoding="utf-8") as f:
    for item in commands:
        f.write(f'# case_id={item["case_id"]} | scenario={item["scenario"]} | gt_hazard={item["gt_hazard_type"]} | gt_severity={item["gt_severity_label"]}\n')
        f.write(item["command"] + "\n\n")

print(f"Command preview written to: {OUTPUT_TXT}")
print(f"Commands created: {len(commands)}")
print(f"Skipped cases: {len(skipped)}")

print("\nCommands by scenario:")
for scenario, count in Counter(item["scenario"] for item in commands).items():
    print(f"  {scenario}: {count}")

print("\nSkipped by scenario:")
for scenario, count in Counter(row["scenario"] for row in skipped).items():
    print(f"  {scenario}: {count}")

print("\nFirst 5 commands:")
for item in commands[:5]:
    print(f'\n# {item["case_id"]} | {item["scenario"]} | {item["gt_severity_label"]}')
    print(item["command"])
