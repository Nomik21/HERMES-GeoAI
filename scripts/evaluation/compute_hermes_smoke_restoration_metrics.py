from pathlib import Path
import csv
import json
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_RUN_RESULTS = PROJECT_ROOT / "evaluation" / "hermes_smoke_final_run_results.csv"

IMAGE_LEVEL_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_restoration_image_level.csv"
SUMMARY_CSV = PROJECT_ROOT / "evaluation" / "hermes_smoke_restoration_metrics.csv"
SUMMARY_TXT = PROJECT_ROOT / "evaluation" / "hermes_smoke_restoration_metrics.txt"


def stems_from_image_paths(image_paths):
    stems = []
    for rel_path in image_paths.split(";"):
        rel_path = rel_path.strip()
        if rel_path:
            stems.append(Path(rel_path).stem)
    return stems


def case_part_from_image_paths(image_paths):
    return "__".join(stems_from_image_paths(image_paths))


def read_json(path):
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def as_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return bool(value)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return ""
    return sum(nums) / len(nums)


def rate(num, den):
    return num / den if den else 0.0


def f4(value):
    if value == "":
        return ""
    return f"{float(value):.4f}"


def normalize_single_gira_result(gira, *, case_id, scenario, image_index, fallback_image, gira_path):
    results = gira.get("results")

    if isinstance(results, list) and results:
        item = results[0]
    else:
        item = gira

    q_before = item.get("Q_before", gira.get("Q_before", ""))
    q_after = item.get("Q_after", gira.get("Q_after", ""))
    delta_q = item.get("delta_Q", gira.get("delta_Q", ""))

    return {
        "case_id": case_id,
        "scenario": scenario,
        "image_index": item.get("image_index", image_index),
        "image": item.get("image", fallback_image),
        "source_type": item.get("source_type", ""),
        "needs_restoration": str(as_bool(item.get("needs_restoration", gira.get("needs_restoration", False)))),
        "accepted": str(as_bool(item.get("accepted", gira.get("accepted", False)))),
        "Q_before": q_before,
        "Q_after": q_after,
        "delta_Q": delta_q,
        "tool_chain": json.dumps(item.get("tool_chain", gira.get("tool_chain", [])), ensure_ascii=False),
        "tool_selector": item.get("tool_selector", gira.get("tool_selector", "")),
        "decision": item.get("decision", gira.get("decision", "")),
        "gira_path": str(gira_path.relative_to(PROJECT_ROOT)),
        "gira_source": "single_image",
    }


def extract_gira_rows(row):
    case_id = row["case_id"]
    scenario = row["scenario"]
    stems = stems_from_image_paths(row["image_paths"])
    case_part = "__".join(stems)

    inter_dir = PROJECT_ROOT / "outputs" / "intermediate"
    pair_gira_path = inter_dir / f"{case_part}_gira.json"
    pair_gira = read_json(pair_gira_path)

    image_rows = []

    # Preferred path: pair-level / collection-level GIRA JSON
    if pair_gira:
        results = pair_gira.get("results")

        if isinstance(results, list) and results:
            for idx, item in enumerate(results):
                image_rows.append({
                    "case_id": case_id,
                    "scenario": scenario,
                    "image_index": item.get("image_index", idx),
                    "image": item.get("image", stems[idx] if idx < len(stems) else ""),
                    "source_type": item.get("source_type", ""),
                    "needs_restoration": str(as_bool(item.get("needs_restoration"))),
                    "accepted": str(as_bool(item.get("accepted"))),
                    "Q_before": item.get("Q_before", ""),
                    "Q_after": item.get("Q_after", ""),
                    "delta_Q": item.get("delta_Q", ""),
                    "tool_chain": json.dumps(item.get("tool_chain", []), ensure_ascii=False),
                    "tool_selector": item.get("tool_selector", ""),
                    "decision": item.get("decision", ""),
                    "gira_path": str(pair_gira_path.relative_to(PROJECT_ROOT)),
                    "gira_source": "pair_level_results",
                })

            return image_rows

        needs = to_list(pair_gira.get("needs_restoration"))
        accepted = to_list(pair_gira.get("accepted"))
        q_before = to_list(pair_gira.get("Q_before"))
        q_after = to_list(pair_gira.get("Q_after"))
        delta_q = to_list(pair_gira.get("delta_Q"))
        source_types = to_list(pair_gira.get("source_types"))

        n = max(
            len(stems),
            len(needs),
            len(accepted),
            len(q_before),
            len(q_after),
            len(delta_q),
            len(source_types),
            1,
        )

        for idx in range(n):
            image_rows.append({
                "case_id": case_id,
                "scenario": scenario,
                "image_index": idx,
                "image": stems[idx] if idx < len(stems) else "",
                "source_type": source_types[idx] if idx < len(source_types) else "",
                "needs_restoration": str(as_bool(needs[idx])) if idx < len(needs) else "",
                "accepted": str(as_bool(accepted[idx])) if idx < len(accepted) else "",
                "Q_before": q_before[idx] if idx < len(q_before) else "",
                "Q_after": q_after[idx] if idx < len(q_after) else "",
                "delta_Q": delta_q[idx] if idx < len(delta_q) else "",
                "tool_chain": "",
                "tool_selector": "",
                "decision": "",
                "gira_path": str(pair_gira_path.relative_to(PROJECT_ROOT)),
                "gira_source": "pair_level_arrays",
            })

        return image_rows

    # Fallback path: individual image-level GIRA JSONs
    for idx, stem in enumerate(stems):
        single_gira_path = inter_dir / f"{stem}_gira.json"
        single_gira = read_json(single_gira_path)

        if single_gira:
            image_rows.append(
                normalize_single_gira_result(
                    single_gira,
                    case_id=case_id,
                    scenario=scenario,
                    image_index=idx,
                    fallback_image=stem,
                    gira_path=single_gira_path,
                )
            )
        else:
            image_rows.append({
                "case_id": case_id,
                "scenario": scenario,
                "image_index": idx,
                "image": stem,
                "source_type": "",
                "needs_restoration": "",
                "accepted": "",
                "Q_before": "",
                "Q_after": "",
                "delta_Q": "",
                "tool_chain": "",
                "tool_selector": "",
                "decision": "Missing pair-level and single-image GIRA JSON.",
                "gira_path": str(single_gira_path.relative_to(PROJECT_ROOT)),
                "gira_source": "missing",
            })

    return image_rows


with FINAL_RUN_RESULTS.open("r", encoding="utf-8-sig", newline="") as f:
    run_rows = list(csv.DictReader(f))

image_rows = []

for row in run_rows:
    image_rows.extend(extract_gira_rows(row))

image_fieldnames = [
    "case_id",
    "scenario",
    "image_index",
    "image",
    "source_type",
    "needs_restoration",
    "accepted",
    "Q_before",
    "Q_after",
    "delta_Q",
    "tool_chain",
    "tool_selector",
    "decision",
    "gira_path",
    "gira_source",
]

with IMAGE_LEVEL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=image_fieldnames)
    writer.writeheader()
    writer.writerows(image_rows)

summary_rows = []


def summarize(group_name, rows):
    total = len(rows)

    needs_count = sum(r["needs_restoration"].lower() == "true" for r in rows)
    accepted_count = sum(r["accepted"].lower() == "true" for r in rows)
    accepted_needed_count = sum(
        r["needs_restoration"].lower() == "true" and r["accepted"].lower() == "true"
        for r in rows
    )

    q_before_values = [as_float(r["Q_before"]) for r in rows]
    q_after_values = [as_float(r["Q_after"]) for r in rows]
    delta_values = [as_float(r["delta_Q"]) for r in rows]

    q_before_values = [v for v in q_before_values if v is not None]
    q_after_values = [v for v in q_after_values if v is not None]
    delta_values = [v for v in delta_values if v is not None]

    return {
        "group": group_name,
        "num_images": total,
        "restoration_needed_count": needs_count,
        "restoration_needed_rate": f4(rate(needs_count, total)),
        "accepted_count": accepted_count,
        "accepted_rate_all_images": f4(rate(accepted_count, total)),
        "accepted_among_needed_count": accepted_needed_count,
        "accepted_among_needed_rate": f4(rate(accepted_needed_count, needs_count)),
        "mean_Q_before": f4(mean(q_before_values)),
        "mean_Q_after": f4(mean(q_after_values)),
        "mean_delta_Q": f4(mean(delta_values)),
    }


summary_rows.append(summarize("all", image_rows))

by_scenario = defaultdict(list)
for row in image_rows:
    by_scenario[row["scenario"]].append(row)

for scenario, rows in by_scenario.items():
    summary_rows.append(summarize(scenario, rows))

with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "group",
        "num_images",
        "restoration_needed_count",
        "restoration_needed_rate",
        "accepted_count",
        "accepted_rate_all_images",
        "accepted_among_needed_count",
        "accepted_among_needed_rate",
        "mean_Q_before",
        "mean_Q_after",
        "mean_delta_Q",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

lines = []
lines.append("HERMES Smoke Restoration Evaluation")
lines.append("=" * 50)

for row in summary_rows:
    lines.append("")
    lines.append(row["group"])
    lines.append("-" * 50)
    lines.append(f"Images: {row['num_images']}")
    lines.append(f"Restoration needed: {row['restoration_needed_count']} / {row['num_images']} = {row['restoration_needed_rate']}")
    lines.append(f"Accepted restorations: {row['accepted_count']} / {row['num_images']} = {row['accepted_rate_all_images']}")
    lines.append(f"Accepted among needed: {row['accepted_among_needed_count']} / {row['restoration_needed_count']} = {row['accepted_among_needed_rate']}")
    lines.append(f"Mean Q_before: {row['mean_Q_before']}")
    lines.append(f"Mean Q_after: {row['mean_Q_after']}")
    lines.append(f"Mean delta_Q: {row['mean_delta_Q']}")

SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Image-level restoration CSV written to: {IMAGE_LEVEL_CSV}")
print(f"Restoration metrics CSV written to: {SUMMARY_CSV}")
print(f"Restoration metrics TXT written to: {SUMMARY_TXT}")

print("\nRows by GIRA source:")
source_counts = defaultdict(int)
for row in image_rows:
    source_counts[row["gira_source"]] += 1

for source, count in source_counts.items():
    print(f"  {source}: {count}")

print("\n" + "\n".join(lines))
