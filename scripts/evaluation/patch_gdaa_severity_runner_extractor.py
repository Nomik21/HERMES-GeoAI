from pathlib import Path

path = Path("scripts/evaluation/run_gdaa_severity_evaluation.py")
text = path.read_text(encoding="utf-8")

# 1) Add top-level overall_severity candidate
text = text.replace(
'''        candidates.extend([
            result.get("severity_label"),''',
'''        candidates.extend([
            result.get("overall_severity"),
            result.get("severity_label"),'''
)

# 2) Add nested overall_severity candidate
text = text.replace(
'''                candidates.extend([
                    sub.get("severity_label"),''',
'''                candidates.extend([
                    sub.get("overall_severity"),
                    sub.get("severity_label"),'''
)

# 3) Improve resume logic: skip only successful rows that already have extracted pred_severity_id
text = text.replace(
'''    completed_success = {
        case_id
        for case_id, row in latest_by_case.items()
        if row.get("status") == "success"
    }''',
'''    completed_success = {
        case_id
        for case_id, row in latest_by_case.items()
        if row.get("status") == "success" and str(row.get("pred_severity_id", "")).strip() != ""
    }'''
)

path.write_text(text, encoding="utf-8")
print(f"Patched: {path}")
