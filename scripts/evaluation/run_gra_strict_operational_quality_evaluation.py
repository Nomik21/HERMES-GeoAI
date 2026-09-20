from pathlib import Path
import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from statistics import mean

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from google import genai
from google.genai import types


PROJECT_ROOT = Path.cwd()

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

ROOT = PROJECT_ROOT / "evaluation" / "gra_report_evaluation_set"
OUT_DIR = ROOT / "strict_operational_evaluation"

DEFAULT_MODEL = os.getenv("GEMINI_GRA_STRICT_EVAL_MODEL", "gemini-3.1-pro-preview")

SCENARIOS = [
    "cross_view_svi_rsi",
    "pre_post_svi",
    "wildfire_building",
    "object_detection",
]

DIMENSIONS = [
    "operational_specificity",
    "evidence_action_alignment",
    "uncertainty_risk_handling",
    "decision_support_value",
]

FIELDNAMES = [
    "run_timestamp",
    "scenario",
    "case_id",
    "report_json_path",
    "status",
    "model",
    "operational_specificity",
    "evidence_action_alignment",
    "uncertainty_risk_handling",
    "decision_support_value",
    "overall_mean",
    "overall_verdict",
    "major_issues",
    "recommended_improvements",
    "operational_specificity_rationale",
    "evidence_action_alignment_rationale",
    "uncertainty_risk_handling_rationale",
    "decision_support_value_rationale",
    "raw_eval_json",
    "error_summary",
]


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


def dict_or_empty(value):
    return value if isinstance(value, dict) else {}


def normalize_eval_block(block, key: str) -> dict:
    if isinstance(block, list):
        block = block[0] if block else {}

    if isinstance(block, dict):
        score = block.get("score")
        rationale = block.get("rationale", "")
    else:
        score = block
        rationale = ""

    if isinstance(score, str):
        match = re.search(r"[1-5]", score)
        if match:
            score = int(match.group(0))

    score = int(score)

    if score < 1 or score > 5:
        raise ValueError(f"Invalid score for {key}: {score}")

    return {
        "score": score,
        "rationale": str(rationale or ""),
    }


def compact_operational_payload(data: dict) -> dict:
    damage = dict_or_empty(data.get("damage_assessment", {}))
    machine = dict_or_empty(data.get("machine_readable_summary", {}))

    return {
        "case_id": data.get("case_id"),
        "report_scenario": data.get("report_scenario"),
        "type_of_disaster": data.get("type_of_disaster"),
        "input_mode": data.get("input_mode"),

        "minimal_structured_context": {
            "analysis_mode": damage.get("analysis_mode") or machine.get("analysis_mode"),
            "damage_presence": damage.get("damage_presence"),
            "severity": damage.get("severity"),
            "overall_severity": damage.get("overall_severity"),
            "three_level_severity": damage.get("three_level_severity"),
            "domain_specific_class": damage.get("domain_specific_class"),
            "wildfire_damage_class": damage.get("wildfire_damage_class"),
            "damage_percent_range": damage.get("damage_percent_range"),
            "confidence": damage.get("confidence"),
            "num_detections": damage.get("num_detections") or machine.get("num_detections"),
            "object_counts": damage.get("object_counts") or machine.get("object_counts"),
            "category_counts": damage.get("category_counts") or machine.get("category_counts"),
        },

        "executive_summary": data.get("executive_summary"),
        "final_report": data.get("final_report"),
    }


def build_prompt(payload: dict) -> str:
    return f"""
You are a strict operational evaluator for disaster-response situational awareness reports.

Evaluate the report as a user-facing operational product.
Do NOT give high scores merely because the report is internally consistent.
Do NOT reward boilerplate wording.
Do NOT require perfect engineering detail, but penalize vague, generic, mismatched, or non-actionable reporting.

You are given:
- minimal structured context
- executive summary
- final report

You are NOT asked to evaluate visual correctness against the image.
You ARE asked to evaluate whether the report is operationally useful, specific, appropriately cautious, and decision-supportive.

Score each dimension from 1 to 5:

1. operational_specificity:
   Does the report use scenario-specific and case-specific language?
   Penalize generic templates, irrelevant disaster wording, irrelevant local file paths, or text that could apply to any case.

2. evidence_action_alignment:
   Are recommended actions aligned with the stated severity, detections, damage state, and scenario?
   Penalize actions that do not make sense for the reported damage state, such as inspecting intact components in a totally destroyed building, or claiming detections provide spatial cues when zero detections exist.

3. uncertainty_risk_handling:
   Does the report clearly communicate limitations, uncertainty, safety caveats, and the need for field verification without overclaiming?
   Penalize overconfidence, missing uncertainty, hidden processing errors, or failure to distinguish rapid assessment from definitive inspection.

4. decision_support_value:
   Would this report help an emergency-management, field-verification, or triage team decide what to do next?
   Penalize lack of prioritization, vague recommendations, unclear asset focus, or absence of actionable next steps.

Return ONLY valid JSON with this exact schema:

{{
  "operational_specificity": {{
    "score": 1,
    "rationale": "..."
  }},
  "evidence_action_alignment": {{
    "score": 1,
    "rationale": "..."
  }},
  "uncertainty_risk_handling": {{
    "score": 1,
    "rationale": "..."
  }},
  "decision_support_value": {{
    "score": 1,
    "rationale": "..."
  }},
  "overall_verdict": "pass | borderline | fail",
  "major_issues": ["..."],
  "recommended_improvements": ["..."]
}}

Report payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def score_report(client, model: str, report_path: Path) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    payload = compact_operational_payload(data)
    prompt = build_prompt(payload)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    parsed = extract_json_object(response.text or "")

    scores = {}
    for key in DIMENSIONS:
        parsed[key] = normalize_eval_block(parsed.get(key, {}), key)
        scores[key] = parsed[key]["score"]

    return {
        "parsed": parsed,
        "overall_mean": mean(scores.values()),
    }


def discover_reports() -> list[Path]:
    report_paths = []

    for scenario in SCENARIOS:
        reports_dir = ROOT / scenario / "reports"
        if not reports_dir.exists():
            continue

        for case_dir in sorted([p for p in reports_dir.iterdir() if p.is_dir()]):
            json_files = sorted(case_dir.glob("*.json"))
            if json_files:
                report_paths.append(json_files[0])

    return report_paths


def load_existing(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_csv = OUT_DIR / "gra_strict_operational_scores.csv"
    raw_jsonl = OUT_DIR / "gra_strict_operational_scores.jsonl"

    reports_all = discover_reports()
    reports = reports_all[: args.limit] if args.limit is not None else reports_all

    existing_rows = load_existing(results_csv)
    done_paths = {
        row["report_json_path"]
        for row in existing_rows
        if row.get("status") == "success"
    }

    rows = existing_rows[:]

    print("HERMES GRA Strict Operational Quality Evaluation")
    print("=" * 80)
    print(f"Reports discovered: {len(reports_all)}")
    print(f"Reports selected: {len(reports)}")
    print(f"Model: {args.model}")
    print(f"Output CSV: {results_csv}")
    print("")

    for idx, report_path in enumerate(reports, start=1):
        rel_path = str(report_path.relative_to(PROJECT_ROOT))

        if args.resume and rel_path in done_paths:
            print(f"[{idx}/{len(reports)}] SKIP success | {rel_path}")
            continue

        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            scenario = data.get("report_scenario") or report_path.parts[-4]
            case_id = data.get("case_id") or report_path.parent.name

            print(f"[{idx}/{len(reports)}] EVAL | {scenario} | {case_id}")

            start = time.time()
            result = score_report(client, args.model, report_path)
            elapsed = time.time() - start

            parsed = result["parsed"]

            row = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "scenario": scenario,
                "case_id": case_id,
                "report_json_path": rel_path,
                "status": "success",
                "model": args.model,
                "operational_specificity": parsed["operational_specificity"]["score"],
                "evidence_action_alignment": parsed["evidence_action_alignment"]["score"],
                "uncertainty_risk_handling": parsed["uncertainty_risk_handling"]["score"],
                "decision_support_value": parsed["decision_support_value"]["score"],
                "overall_mean": f"{result['overall_mean']:.4f}",
                "overall_verdict": parsed.get("overall_verdict", ""),
                "major_issues": json.dumps(parsed.get("major_issues", []), ensure_ascii=False),
                "recommended_improvements": json.dumps(parsed.get("recommended_improvements", []), ensure_ascii=False),
                "operational_specificity_rationale": parsed["operational_specificity"].get("rationale", ""),
                "evidence_action_alignment_rationale": parsed["evidence_action_alignment"].get("rationale", ""),
                "uncertainty_risk_handling_rationale": parsed["uncertainty_risk_handling"].get("rationale", ""),
                "decision_support_value_rationale": parsed["decision_support_value"].get("rationale", ""),
                "raw_eval_json": json.dumps(parsed, ensure_ascii=False),
                "error_summary": "",
            }

            rows.append(row)

            with raw_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            save_rows(results_csv, rows)

            print(
                f"  scores: OS={row['operational_specificity']} | "
                f"EAA={row['evidence_action_alignment']} | "
                f"URH={row['uncertainty_risk_handling']} | "
                f"DSV={row['decision_support_value']} | "
                f"mean={row['overall_mean']} | "
                f"verdict={row['overall_verdict']} | "
                f"{elapsed:.2f}s"
            )

        except Exception as exc:
            error = str(exc)
            print(f"  ERROR: {error}")

            row = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "scenario": report_path.parts[-4] if len(report_path.parts) >= 4 else "",
                "case_id": report_path.parent.name,
                "report_json_path": rel_path,
                "status": "failed",
                "model": args.model,
                "operational_specificity": "",
                "evidence_action_alignment": "",
                "uncertainty_risk_handling": "",
                "decision_support_value": "",
                "overall_mean": "",
                "overall_verdict": "",
                "major_issues": "",
                "recommended_improvements": "",
                "operational_specificity_rationale": "",
                "evidence_action_alignment_rationale": "",
                "uncertainty_risk_handling_rationale": "",
                "decision_support_value_rationale": "",
                "raw_eval_json": "",
                "error_summary": error,
            }

            rows.append(row)
            save_rows(results_csv, rows)

    print("")
    print("DONE ✅")
    print(f"Scores CSV: {results_csv}")
    print(f"Raw JSONL: {raw_jsonl}")


if __name__ == "__main__":
    main()
