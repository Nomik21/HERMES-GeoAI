from pathlib import Path, PureWindowsPath
import csv
import shutil
import re
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_CSV = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set" / "gra_report_eval_10_object_detection_run_results.csv"
OUT_ROOT = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set"
SOURCE_REPORT_DIR = PROJECT_ROOT / "outputs" / "Scenarios" / "object_detection" / "reports"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "case"


def image_stem_from_row(row):
    image_paths = str(row.get("image_paths", "")).strip()
    first_path = image_paths.split(";")[0].strip()
    return PureWindowsPath(first_path).stem


def main():
    rows, fieldnames = read_csv(RESULTS_CSV)

    copied_files = 0
    repaired_rows = 0
    missing = []

    for row in rows:
        case_id = row["case_id"]
        image_stem = image_stem_from_row(row)

        src_json = SOURCE_REPORT_DIR / f"object_detection__{image_stem}_report.json"
        src_md = SOURCE_REPORT_DIR / f"object_detection__{image_stem}_report.md"
        src_html = SOURCE_REPORT_DIR / f"object_detection__{image_stem}_report.html"

        src_files = [src_json, src_md, src_html]

        if not src_json.exists():
            missing.append(str(src_json.relative_to(PROJECT_ROOT)))
            continue

        dst_dir = OUT_ROOT / "object_detection" / "reports" / safe_name(case_id)
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src in src_files:
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
                copied_files += 1
            else:
                missing.append(str(src.relative_to(PROJECT_ROOT)))

        row["status"] = "success"
        row["report_exists"] = "True"
        row["report_json_path"] = str(src_json.relative_to(PROJECT_ROOT))
        row["report_md_path"] = str(src_md.relative_to(PROJECT_ROOT))
        row["report_html_path"] = str(src_html.relative_to(PROJECT_ROOT))
        row["error_summary"] = ""

        repaired_rows += 1

    write_csv(RESULTS_CSV, rows, fieldnames)

    print(f"Repaired object_detection rows: {repaired_rows}")
    print(f"Copied report files: {copied_files}")
    print(f"Missing files: {len(missing)}")

    if missing:
        print("")
        print("Missing examples:")
        for item in missing[:20]:
            print(f"  {item}")

    print("")
    print("Object-detection reports copied to:")
    print(OUT_ROOT / "object_detection" / "reports")
    print("DONE ✅")


if __name__ == "__main__":
    main()
