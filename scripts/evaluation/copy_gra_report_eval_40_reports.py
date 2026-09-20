from pathlib import Path
import csv
import shutil
import re
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_RESULTS = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set" / "gra_report_eval_40_run_results.csv"
OUT_ROOT = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_name(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "case"


def latest_by_case(rows):
    latest = {}
    for row in rows:
        latest[(row["scenario"], row["case_id"])] = row
    return list(latest.values())


def main():
    rows = read_csv(RUN_RESULTS)
    rows = latest_by_case(rows)

    copied = 0
    missing = []
    copied_cases = defaultdict(int)

    for row in rows:
        if row.get("status") != "success":
            continue

        scenario = row["scenario"]
        case_id = row["case_id"]

        src_json = PROJECT_ROOT / row["report_json_path"]
        src_md = src_json.with_suffix(".md")
        src_html = src_json.with_suffix(".html")

        dst_dir = OUT_ROOT / scenario / "reports" / safe_name(case_id)
        dst_dir.mkdir(parents=True, exist_ok=True)

        case_copied = 0

        for src in [src_json, src_md, src_html]:
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
                copied += 1
                case_copied += 1
            else:
                missing.append(str(src.relative_to(PROJECT_ROOT)))

        if case_copied > 0:
            copied_cases[scenario] += 1

    print(f"Copied report files: {copied}")
    print(f"Missing report files: {len(missing)}")
    print("")
    print("Copied cases by scenario:")

    for scenario in ["cross_view_svi_rsi", "pre_post_svi", "wildfire_building", "object_detection"]:
        print(f"  {scenario}: {copied_cases[scenario]}")

    if missing:
        print("")
        print("Missing examples:")
        for item in missing[:20]:
            print(f"  {item}")

    print("")
    print(f"Reports copied under: {OUT_ROOT}")
    print("DONE ✅")


if __name__ == "__main__":
    main()
