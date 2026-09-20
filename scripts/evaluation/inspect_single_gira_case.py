from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_RUN_RESULTS = PROJECT_ROOT / "evaluation" / "hermes_smoke_final_run_results.csv"

TARGET_CASE_ID = "115310275000667_vs_1197746641332199(1)"


def case_part_from_image_paths(image_paths):
    stems = []
    for rel_path in image_paths.split(";"):
        rel_path = rel_path.strip()
        if rel_path:
            stems.append(Path(rel_path).stem)
    return "__".join(stems)


def read_json(path):
    if not path.exists():
        print(f"Missing JSON: {path}")
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def length_of(value):
    if isinstance(value, list):
        return len(value)
    if value is None:
        return 0
    return 1


with FINAL_RUN_RESULTS.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

target = None
for row in rows:
    if row["case_id"] == TARGET_CASE_ID:
        target = row
        break

if target is None:
    raise SystemExit(f"Case not found: {TARGET_CASE_ID}")

case_part = case_part_from_image_paths(target["image_paths"])
gira_path = PROJECT_ROOT / "outputs" / "intermediate" / f"{case_part}_gira.json"

print("=" * 100)
print("TARGET CASE")
print("=" * 100)
print(f"case_id: {target['case_id']}")
print(f"scenario: {target['scenario']}")
print(f"image_paths: {target['image_paths']}")
print(f"case_part: {case_part}")
print(f"gira_path: {gira_path}")
print(f"gira_exists: {gira_path.exists()}")

gira = read_json(gira_path)

print("\n" + "=" * 100)
print("GIRA TOP-LEVEL KEYS")
print("=" * 100)
print(list(gira.keys()))

fields = [
    "input_paths",
    "source_types",
    "final_paths",
    "restored_paths",
    "accepted",
    "needs_restoration",
    "Q_before",
    "Q_after",
    "delta_Q",
    "results",
]

print("\n" + "=" * 100)
print("FIELD LENGTHS")
print("=" * 100)

for field in fields:
    value = gira.get(field)
    print(f"{field}: type={type(value).__name__}, length={length_of(value)}")

print("\n" + "=" * 100)
print("FIELD VALUES")
print("=" * 100)

for field in fields:
    value = gira.get(field)
    print(f"\n{field}:")
    print(json.dumps(value, ensure_ascii=False, indent=2)[:4000])
