from pathlib import Path
import json
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSONL = PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_pilot_results.jsonl"

def walk(obj, path="result", hits=None):
    if hits is None:
        hits = []

    keywords = [
        "severity",
        "damage",
        "level",
        "class",
        "rating",
        "assessment",
        "prediction",
        "label",
    ]

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}"
            key_hit = any(word in str(k).lower() for word in keywords)

            if isinstance(v, (str, int, float, bool)) or v is None:
                val_text = str(v)
                val_hit = any(word in val_text.lower() for word in keywords)
                if key_hit or val_hit:
                    hits.append((new_path, v))
            else:
                walk(v, new_path, hits)

    elif isinstance(obj, list):
        for i, item in enumerate(obj[:10]):
            walk(item, f"{path}[{i}]", hits)

    return hits

rows = []
with JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"JSONL: {JSONL}")
print(f"Rows loaded: {len(rows)}")

for i, row in enumerate(rows[:3], start=1):
    print("\n" + "=" * 100)
    print(f"CASE {i}: {row.get('case_id')}")
    print(f"Scenario: {row.get('scenario')}")
    print(f"GT: {row.get('gt_severity_label')} / {row.get('gt_severity_id')}")

    result = row.get("result", {})

    print("\nTop-level result keys:")
    if isinstance(result, dict):
        print(list(result.keys()))
    else:
        print(type(result))

    print("\nSeverity/damage/class-related paths:")
    hits = walk(result)
    for path, value in hits[:120]:
        print(f"  {path}: {value}")

    print("\nCompact result preview:")
    pprint(result, depth=3, width=160)
