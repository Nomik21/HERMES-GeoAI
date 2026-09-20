from pathlib import Path
import csv
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_CSV = PROJECT_ROOT / "evaluation" / "hermes_dataset_manifest.csv"
OBJECT_GT_CSV = PROJECT_ROOT / "evaluation" / "hermes_object_groundtruth.csv"

with MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    manifest_rows = list(csv.DictReader(f))

with OBJECT_GT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    object_rows = list(csv.DictReader(f))

manifest_keys = {
    (row["scenario"], row["case_id"])
    for row in manifest_rows
}

object_keys = {
    (row["scenario"], row["case_id"])
    for row in object_rows
}

missing_in_manifest = sorted(object_keys - manifest_keys)

manifest_object_candidates = {
    (row["scenario"], row["case_id"])
    for row in manifest_rows
    if row["scenario"] in {"cross_view_svi_rsi", "pre_post_svi"}
}

missing_object_gt = sorted(manifest_object_candidates - object_keys)

print(f"Manifest rows: {len(manifest_rows)}")
print(f"Object GT rows: {len(object_rows)}")
print(f"Object GT keys: {len(object_keys)}")
print(f"Object-capable manifest keys: {len(manifest_object_candidates)}")

print(f"\nObject GT rows missing in manifest: {len(missing_in_manifest)}")
if missing_in_manifest:
    for item in missing_in_manifest[:20]:
        print("  ", item)

print(f"\nManifest object-capable cases missing object GT: {len(missing_object_gt)}")
if missing_object_gt:
    for item in missing_object_gt[:20]:
        print("  ", item)

print("\nObject GT rows by scenario:")
for scenario, count in Counter(row["scenario"] for row in object_rows).items():
    print(f"  {scenario}: {count}")

if not missing_in_manifest and not missing_object_gt:
    print("\nOBJECT GROUND-TRUTH JOIN VALIDATION PASSED ✅")
else:
    print("\nOBJECT GROUND-TRUTH JOIN VALIDATION FAILED ❌")
