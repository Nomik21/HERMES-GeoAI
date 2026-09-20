from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_CSVS = [
    PROJECT_ROOT / "evaluation" / "hermes_smoke_final_run_results.csv",
    PROJECT_ROOT / "evaluation" / "hermes_smoke_predictions.csv",
    PROJECT_ROOT / "evaluation" / "hermes_smoke_results.csv",
    PROJECT_ROOT / "evaluation" / "hermes_gdaa_severity_full_results.csv",
]

SEARCH_DIRS = [
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "evaluation",
]

REPORT_KEYWORDS = [
    "report",
    "gra",
    "reasoning",
    "html",
    "markdown",
    "md_path",
    "txt_path",
    "output",
]

REPORT_EXTENSIONS = {".html", ".md", ".txt", ".json"}


def read_csv_rows(path):
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []


def looks_like_report_column(col):
    c = col.lower()
    return any(k in c for k in REPORT_KEYWORDS)


def resolve_possible_path(value):
    value = str(value or "").strip().strip('"')

    if not value:
        return None

    p = Path(value)

    if p.is_absolute():
        return p

    return PROJECT_ROOT / p


def main():
    print("HERMES GRA Report Source Inspection")
    print("=" * 90)
    print(f"Project root: {PROJECT_ROOT}")
    print("")

    for csv_path in CANDIDATE_CSVS:
        print("-" * 90)
        print(f"CSV: {csv_path}")
        print(f"Exists: {csv_path.exists()}")

        rows, fieldnames = read_csv_rows(csv_path)

        print(f"Rows: {len(rows)}")
        print(f"Columns: {fieldnames}")
        print("")

        if not rows:
            continue

        report_cols = [c for c in fieldnames if looks_like_report_column(c)]

        print(f"Report-like columns: {report_cols}")

        existing_paths = []

        for col in report_cols:
            for row in rows:
                p = resolve_possible_path(row.get(col, ""))

                if p and p.exists() and p.is_file():
                    existing_paths.append((col, str(p)))

        unique_existing = []
        seen = set()

        for item in existing_paths:
            if item[1] not in seen:
                unique_existing.append(item)
                seen.add(item[1])

        print(f"Existing report-like file paths found from CSV: {len(unique_existing)}")

        for col, path in unique_existing[:30]:
            print(f"  {col}: {path}")

        print("")

    print("=" * 90)
    print("Filesystem scan for possible report files")
    print("=" * 90)

    found = []

    for root in SEARCH_DIRS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            name = path.name.lower()

            if suffix not in REPORT_EXTENSIONS:
                continue

            if any(k in name for k in ["report", "gra", "reasoning"]):
                found.append(path)

    print(f"Possible report files found: {len(found)}")

    for path in found[:80]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
