from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_RESULTS_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_run_results.csv"

TARGET_SCENARIOS = [
    "pre_post_svi",
    "wildfire_building",
]

KEYWORDS = [
    "severity",
    "damage",
    "class",
    "wildfire",
    "overall",
    "confidence",
    "assessment",
]


def read_json(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def extract_case_part(image_paths):
    stems = []
    for rel_path in image_paths.split(";"):
        rel_path = rel_path.strip()
        if rel_path:
            stems.append(Path(rel_path).stem)
    return "__".join(stems)


def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k

            key_hit = any(word.lower() in k.lower() for word in KEYWORDS)

            if key_hit and not isinstance(v, (dict, list)):
                print(f"  {path}: {v}")

            if isinstance(v, (dict, list)):
                walk(v, path)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            walk(item, f"{prefix}[{i}]")


def print_top_level(title, data):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if isinstance(data, dict):
        print("Top-level keys:")
        print("  " + ", ".join(data.keys()))
    else:
        print(type(data))

    print("\nRelevant fields:")
    walk(data)


with RUN_RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

selected = []

for scenario in TARGET_SCENARIOS:
    for row in rows:
        if row["scenario"] == scenario and row["status"] == "success":
            selected.append(row)
            break

for row in selected:
    scenario = row["scenario"]
    case_id = row["case_id"]
    case_part = extract_case_part(row["image_paths"])

    report_path = PROJECT_ROOT / row["report_json_path"]
    gdaa_path = PROJECT_ROOT / "outputs" / "intermediate" / f"{case_part}_gdaa.json"
    gra_path = PROJECT_ROOT / "outputs" / "intermediate" / f"{case_part}_gra.json"

    print("\n\n" + "#" * 120)
    print(f"CASE: {case_id}")
    print(f"SCENARIO: {scenario}")
    print(f"GT SEVERITY: {row['gt_severity_label']}")
    print(f"REPORT: {report_path}")
    print(f"GDAA: {gdaa_path}")
    print(f"GRA: {gra_path}")
    print("#" * 120)

    print_top_level("GDAA INTERMEDIATE", read_json(gdaa_path))
    print_top_level("FINAL REPORT JSON", read_json(report_path))
    print_top_level("GRA INTERMEDIATE", read_json(gra_path))
