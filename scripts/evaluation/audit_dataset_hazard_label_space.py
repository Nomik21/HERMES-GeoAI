from pathlib import Path
import csv
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "hermes_benchmark"
EVAL_ROOT = PROJECT_ROOT / "evaluation"

TARGET_TERMS = {"tornado", "landslide", "other"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def norm(text):
    return str(text or "").strip().lower().replace("-", "_").replace(" ", "_")


def image_count_recursive(folder):
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


print("=" * 100)
print("DATASET ROOT")
print("=" * 100)
print(DATA_ROOT)
print(f"exists: {DATA_ROOT.exists()}")

if not DATA_ROOT.exists():
    raise SystemExit("Dataset root not found.")


# --------------------------------------------------------------------------------------
# 1. Dataset folder audit
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("1. DATASET FOLDER STRUCTURE SUMMARY")
print("=" * 100)

for dataset_dir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()]):
    print(f"\nDATASET: {dataset_dir.name}")
    first_level_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

    if not first_level_dirs:
        print("  No class folders found.")
        continue

    for child in first_level_dirs:
        print(f"  {child.name}: {image_count_recursive(child)} image files")


# --------------------------------------------------------------------------------------
# 2. Exact folder/component search for tornado/landslide/other
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("2. EXACT PATH-COMPONENT SEARCH FOR tornado / landslide / other")
print("=" * 100)

component_hits = []

for p in DATA_ROOT.rglob("*"):
    rel = p.relative_to(PROJECT_ROOT)
    components = [norm(part) for part in rel.parts]

    for term in TARGET_TERMS:
        if term in components:
            component_hits.append({
                "term": term,
                "type": "dir" if p.is_dir() else "file",
                "path": str(rel),
            })

if not component_hits:
    print("No exact path-component hits found ✅")
else:
    print(f"Found {len(component_hits)} exact path-component hits:")
    for hit in component_hits[:200]:
        print(f"  {hit['term']} | {hit['type']} | {hit['path']}")
    if len(component_hits) > 200:
        print(f"  ... truncated, remaining: {len(component_hits) - 200}")


# --------------------------------------------------------------------------------------
# 3. Directory-name contains search
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("3. DIRECTORY-NAME CONTAINS SEARCH")
print("=" * 100)

contains_hits = []

for d in DATA_ROOT.rglob("*"):
    if not d.is_dir():
        continue

    name = norm(d.name)

    for term in TARGET_TERMS:
        if term in name:
            contains_hits.append({
                "term": term,
                "name": d.name,
                "path": str(d.relative_to(PROJECT_ROOT)),
            })

if not contains_hits:
    print("No directory-name contains hits found ✅")
else:
    print(f"Found {len(contains_hits)} directory-name contains hits:")
    for hit in contains_hits[:200]:
        print(f"  {hit['term']} | {hit['name']} | {hit['path']}")
    if len(contains_hits) > 200:
        print(f"  ... truncated, remaining: {len(contains_hits) - 200}")


# --------------------------------------------------------------------------------------
# 4. Manifest label audit
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. MANIFEST LABEL AUDIT")
print("=" * 100)

manifest_files = [
    EVAL_ROOT / "hermes_dataset_manifest.csv",
    EVAL_ROOT / "hermes_dataset_manifest_smoke.csv",
    EVAL_ROOT / "hermes_gpa_hazard_smoke_neutral_manifest.csv",
]

for manifest_path in manifest_files:
    print(f"\nMANIFEST: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"exists: {manifest_path.exists()}")

    if not manifest_path.exists():
        continue

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"rows: {len(rows)}")

    if not rows:
        continue

    fieldnames = rows[0].keys()

    for col in ["dataset", "scenario", "gt_hazard_type", "gt_severity_label", "gt_pred_mode"]:
        if col not in fieldnames:
            continue

        counts = Counter(norm(row.get(col, "")) for row in rows)
        print(f"\n  {col}:")
        for label, count in sorted(counts.items()):
            print(f"    {label}: {count}")

    suspicious_rows = []

    for row in rows:
        for col in ["gt_hazard_type", "gt_severity_label", "image_paths", "original_image_paths"]:
            if col not in fieldnames:
                continue

            value = norm(row.get(col, ""))

            for term in TARGET_TERMS:
                if term in value:
                    suspicious_rows.append({
                        "term": term,
                        "column": col,
                        "case_id": row.get("case_id", ""),
                        "value": row.get(col, ""),
                    })

    print("\n  suspicious term hits in manifest:")
    if not suspicious_rows:
        print("    none ✅")
    else:
        for hit in suspicious_rows[:100]:
            print(
                f"    {hit['term']} | {hit['column']} | "
                f"case_id={hit['case_id']} | value={hit['value']}"
            )
        if len(suspicious_rows) > 100:
            print(f"    ... truncated, remaining: {len(suspicious_rows) - 100}")


# --------------------------------------------------------------------------------------
# 5. Final conclusion helper
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("5. DECISION HELPER")
print("=" * 100)

manifest_main = EVAL_ROOT / "hermes_dataset_manifest.csv"

if manifest_main.exists():
    with manifest_main.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    gpa_rows = [
        row for row in rows
        if row.get("scenario") == "single_svi_hazard_classification"
    ]

    gpa_hazards = Counter(norm(row.get("gt_hazard_type", "")) for row in gpa_rows)

    print("GPA hazard evaluation labels from hermes_dataset_manifest.csv:")
    for label, count in sorted(gpa_hazards.items()):
        print(f"  {label}: {count}")

    extra = sorted(set(gpa_hazards.keys()) & TARGET_TERMS)

    print("\nExtra target terms present in GPA gt_hazard_type:")
    if extra:
        print("  " + ", ".join(extra))
        print("\nDECISION: Do NOT use six-class closed set until label space is revised ❌")
    else:
        print("  none ✅")
        print("\nDECISION: GPA benchmark label space is six-class closed set ✅")
        print("Allowed labels: drought, earthquake, flood, hurricane, ice_storm, wildfire")

print("\nDATASET HAZARD LABEL AUDIT COMPLETE ✅")

