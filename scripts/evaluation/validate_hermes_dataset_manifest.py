from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"

missing = []
total_paths = 0

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        case_id = row["case_id"]
        scenario = row["scenario"]
        image_paths = [p for p in row["image_paths"].split(";") if p.strip()]

        for rel_path in image_paths:
            total_paths += 1
            abs_path = PROJECT_ROOT / rel_path

            if not abs_path.exists():
                missing.append({
                    "case_id": case_id,
                    "scenario": scenario,
                    "missing_path": rel_path,
                })

print(f"Manifest: {MANIFEST}")
print(f"Total image paths checked: {total_paths}")
print(f"Missing paths: {len(missing)}")

if missing:
    print("\nFirst missing paths:")
    for item in missing[:20]:
        print(item)
else:
    print("PATH VALIDATION PASSED ✅")
