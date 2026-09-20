from pathlib import Path

path = Path("scripts/evaluation/run_gdaa_severity_evaluation.py")
text = path.read_text(encoding="utf-8")

old = '''def extract_pred_severity(result, scenario):
    """
    Robust extraction from possible HERMES GDAA schemas.
    """
    candidates = []

    if isinstance(result, dict):
        candidates.extend([
            result.get("overall_severity"),
            result.get("severity_label"),
            result.get("predicted_severity"),
            result.get("damage_severity"),
            result.get("severity"),
            result.get("damage_level"),
            result.get("wildfire_damage_class"),
            result.get("damage_class"),
            result.get("class_label"),
            result.get("predicted_class"),
        ])

        for key in [
            "classification",
            "severity_assessment",
            "damage_assessment",
            "building_damage",
            "summary",
            "result",
            "gdaa_output",
        ]:
            sub = result.get(key)
            if isinstance(sub, dict):
                candidates.extend([
                    sub.get("overall_severity"),
                    sub.get("severity_label"),
                    sub.get("predicted_severity"),
                    sub.get("damage_severity"),
                    sub.get("severity"),
                    sub.get("damage_level"),
                    sub.get("wildfire_damage_class"),
                    sub.get("damage_class"),
                    sub.get("class_label"),
                    sub.get("predicted_class"),
                ])

    for item in candidates:
        if item is not None and str(item).strip():
            label = str(item).strip()
            return label, severity_id_from_label(label, scenario)

    return "", ""
'''

new = '''def extract_pred_severity(result, scenario):
    """
    Robust extraction from HERMES GDAA schemas.

    For wildfire_building, the evaluation target is the canonical
    five-class wildfire_damage_class, not the derived overall_severity.
    """
    if not isinstance(result, dict):
        return "", ""

    candidates = []

    if scenario == "wildfire_building":
        candidates.extend([
            result.get("wildfire_damage_class"),
            result.get("Predicted_Class"),
            result.get("predicted_class"),
            result.get("damage_class"),
            result.get("class_label"),
            result.get("overall_severity"),
        ])
    else:
        candidates.extend([
            result.get("overall_severity"),
            result.get("Predicted_Severity"),
            result.get("predicted_severity"),
            result.get("severity_label"),
            result.get("damage_severity"),
            result.get("severity"),
            result.get("damage_level"),
            result.get("damage_class"),
            result.get("class_label"),
            result.get("predicted_class"),
        ])

    for key in [
        "classification",
        "severity_assessment",
        "damage_assessment",
        "building_damage",
        "summary",
        "result",
        "gdaa_output",
    ]:
        sub = result.get(key)
        if isinstance(sub, dict):
            if scenario == "wildfire_building":
                candidates.extend([
                    sub.get("wildfire_damage_class"),
                    sub.get("Predicted_Class"),
                    sub.get("predicted_class"),
                    sub.get("damage_class"),
                    sub.get("class_label"),
                    sub.get("overall_severity"),
                ])
            else:
                candidates.extend([
                    sub.get("overall_severity"),
                    sub.get("Predicted_Severity"),
                    sub.get("predicted_severity"),
                    sub.get("severity_label"),
                    sub.get("damage_severity"),
                    sub.get("severity"),
                    sub.get("damage_level"),
                    sub.get("damage_class"),
                    sub.get("class_label"),
                    sub.get("predicted_class"),
                ])

    for item in candidates:
        if item is not None and str(item).strip():
            label = str(item).strip()

            # Avoid treating derived wildfire "None" as an unrecognized class
            # when a canonical wildfire_damage_class is absent.
            if scenario == "wildfire_building" and label.lower() == "none":
                return "0_No_Damage", 0

            return label, severity_id_from_label(label, scenario)

    return "", ""
'''

if old not in text:
    raise SystemExit("Original extract_pred_severity block not found. Do not patch blindly.")

text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print(f"Patched wildfire-aware extractor in: {path}")
