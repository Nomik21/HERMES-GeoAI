import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

from hermes.core.report_renderer import render_reasoning_report_html


load_dotenv()

GEMINI_REASONING_MODEL = os.getenv("GEMINI_REASONING_MODEL", "gemini-2.5-flash")


def parse_json_like(value: Any) -> Dict[str, Any]:
    """
    Parse dict, JSON string, or markdown-fenced JSON into a dictionary.
    Returns {} if parsing fails.
    """
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        text = str(value)
    else:
        text = value

    text = text.strip()

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}

    return {}


def _safe_get(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _split_list_like(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        if ";" in value:
            return [item.strip() for item in value.split(";") if item.strip()]
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        if value.strip():
            return [value.strip()]

    return []


def _path_for_source_type(
    paths: Any,
    source_types: Any,
    wanted_source_type: str,
) -> Optional[str]:
    path_list = _split_list_like(paths)
    type_list = [
        str(item).strip().upper()
        for item in _split_list_like(source_types)
    ]

    wanted = wanted_source_type.strip().upper()

    for path, source_type in zip(path_list, type_list):
        if source_type == wanted:
            return path

    # Filename fallback
    for path in path_list:
        path_text = str(path).lower()

        if wanted == "RS" and (
            "satellite" in path_text
            or "rsi" in path_text
            or "_rs" in path_text
        ):
            return path

        if wanted == "SV" and (
            "svi" in path_text
            or "street" in path_text
            or "_sv" in path_text
        ):
            return path

    return None


def _normalize_bool(value: Any) -> Optional[bool]:
    """
    Normalize scalar or list-like boolean values.

    For collection cases:
    - [True, True] -> True
    - [False, False] -> False
    - [True, False] -> None, meaning mixed / partial status
    """
    if isinstance(value, list):
        normalized_values = [
            _normalize_bool(item)
            for item in value
            if _normalize_bool(item) is not None
        ]

        if not normalized_values:
            return None

        if all(normalized_values):
            return True

        if not any(normalized_values):
            return False

        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()
        if text in ["true", "yes", "1"]:
            return True
        if text in ["false", "no", "0"]:
            return False

    return None


def extract_perception_summary(gpa_report: Dict[str, Any]) -> Dict[str, Any]:
    mode_and_hazard = gpa_report.get("mode_and_hazard", {})

    return {
        "pred_mode": mode_and_hazard.get("pred_mode"),
        "mode_confidence": mode_and_hazard.get("mode_confidence"),
        "hazard_type": mode_and_hazard.get("hazard_type"),
        "hazard_confidence": mode_and_hazard.get("hazard_confidence"),
        "needs_restoration": mode_and_hazard.get("needs_restoration"),
        "view_types": _safe_get(gpa_report, ["restoration", "view_types"], []),
        "should_invoke_restoration": _safe_get(
            gpa_report,
            ["tool_plan", "should_invoke_restoration"],
            None,
        ),
        "should_invoke_damage_recognition": _safe_get(
            gpa_report,
            ["tool_plan", "should_invoke_damage_recognition"],
            None,
        ),
        "reasoning_text": gpa_report.get("reasoning_text"),
    }


def extract_restoration_summary(gira_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract restoration information from either:
    - single-image GIRA output
    - multi-image / collection-level GIRA output

    Collection-level GIRA outputs contain:
    final_paths, restored_paths, accepted, needs_restoration, results.
    """

    results = gira_report.get("results")

    if not isinstance(results, list):
        results = []

    q_before = gira_report.get("Q_before")
    q_after = gira_report.get("Q_after")
    delta_q = gira_report.get("delta_Q")
    decision = gira_report.get("decision")

    if results:
        if q_before is None:
            q_before = [
                item.get("Q_before")
                for item in results
                if isinstance(item, dict)
            ]

        if q_after is None:
            q_after = [
                item.get("Q_after")
                for item in results
                if isinstance(item, dict)
            ]

        if delta_q is None:
            delta_q = [
                item.get("delta_Q")
                for item in results
                if isinstance(item, dict)
            ]

        if decision is None:
            decisions = [
                item.get("decision")
                for item in results
                if isinstance(item, dict) and item.get("decision")
            ]

            unique_decisions = list(dict.fromkeys(decisions))

            if len(unique_decisions) == 1:
                decision = unique_decisions[0]
            elif len(unique_decisions) > 1:
                decision = " | ".join(unique_decisions)

    return {
        "needs_restoration": gira_report.get("needs_restoration"),
        "accepted": gira_report.get("accepted"),
        "final_path": gira_report.get("final_path"),
        "final_paths": gira_report.get("final_paths"),
        "restored_path": gira_report.get("restored_path"),
        "restored_paths": gira_report.get("restored_paths"),
        "Q_before": q_before,
        "Q_after": q_after,
        "delta_Q": delta_q,
        "decision": decision,
        "results": results,
        "num_images": gira_report.get("num_images"),
        "mode": gira_report.get("mode"),
        "source_types": gira_report.get("source_types"),
        "input_paths": gira_report.get("input_paths"),
    }


def extract_damage_summary(gdaa_report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "analysis_mode": gdaa_report.get("analysis_mode"),
        "pair_id": gdaa_report.get("pair_id"),
        "source_type": gdaa_report.get("source_type"),
        "source_types": gdaa_report.get("source_types"),
        "hazard_type": gdaa_report.get("hazard_type"),

        "damage_presence": gdaa_report.get("damage_presence"),
        "overall_severity": gdaa_report.get("overall_severity"),
        "three_level_severity": (
            gdaa_report.get("three_level_severity")
            or gdaa_report.get("overall_severity")
        ),

        "domain_specific_class": (
            gdaa_report.get("wildfire_damage_class")
            or gdaa_report.get("domain_specific_class")
            or gdaa_report.get("damage_class")
        ),
        "wildfire_damage_class": gdaa_report.get("wildfire_damage_class"),
        "damage_percent_range": gdaa_report.get("damage_percent_range"),
        "confidence": gdaa_report.get("confidence"),

        "affected_assets": (
            gdaa_report.get("affected_assets")
            or list((gdaa_report.get("category_counts") or {}).keys())
        ),

        "objects_detected": (
            gdaa_report.get("objects_detected")
            or gdaa_report.get("object_counts")
        ),

        "detections": gdaa_report.get("detections"),
        "num_detections": gdaa_report.get("num_detections"),
        "object_counts": gdaa_report.get("object_counts"),
        "category_counts": gdaa_report.get("category_counts"),
        "spatial_summary": gdaa_report.get("spatial_summary"),
        "image_path": gdaa_report.get("image_path"),

        "change_evidence": gdaa_report.get("change_evidence"),

        "reasoning": gdaa_report.get("reasoning"),
        "confidence_calibration_notes": gdaa_report.get("confidence_calibration_notes"),
        "downstream_notes": gdaa_report.get("downstream_notes"),

        "pre_image_path": gdaa_report.get("pre_image_path"),
        "post_image_path": gdaa_report.get("post_image_path"),
        "rs_image_path": (
            gdaa_report.get("rs_image_path")
            or gdaa_report.get("rsi_image_path")
            or gdaa_report.get("satellite_image_path")
            or gdaa_report.get("remote_sensing_image_path")
        ),
        "sv_image_path": (
            gdaa_report.get("sv_image_path")
            or gdaa_report.get("svi_image_path")
            or gdaa_report.get("street_view_image_path")
        ),  
        "annotated_image_path": gdaa_report.get("annotated_image_path"),
    }


def infer_final_severity(damage: Dict[str, Any]) -> Optional[str]:
    if damage.get("three_level_severity"):
        return damage["three_level_severity"]

    if damage.get("overall_severity"):
        return damage["overall_severity"]

    wildfire_class = damage.get("wildfire_damage_class")
    if wildfire_class:
        mapping = {
            "0_No_Damage": "None",
            "1_Affected_1_9": "Mild",
            "2_Minor_10_25": "Moderate",
            "3_Major_26_50": "Severe",
            "4_Destroyed_50plus": "Severe",
        }
        return mapping.get(wildfire_class, "Unknown")

    return None


def build_object_recognition_summary(gdaa_report: Dict[str, Any]) -> str:
    object_counts = gdaa_report.get("object_counts")
    objects_detected = gdaa_report.get("objects_detected")
    affected_assets = gdaa_report.get("affected_assets")

    if isinstance(object_counts, dict) and object_counts:
        return ", ".join(f"{k}: {v}" for k, v in object_counts.items())

    if isinstance(objects_detected, dict) and objects_detected:
        active = [k for k, v in objects_detected.items() if v]
        if active:
            return ", ".join(active)

    if isinstance(affected_assets, list) and affected_assets:
        return ", ".join(str(x) for x in affected_assets)

    indicators = gdaa_report.get("building_damage_indicators")
    if isinstance(indicators, dict):
        active = [k for k, v in indicators.items() if v]
        if active:
            return ", ".join(active)

    return "No explicit object-recognition summary available."


def build_evidence_chain(
    perception: Dict[str, Any],
    restoration: Dict[str, Any],
    damage: Dict[str, Any],
    gdaa_report: Dict[str, Any],
) -> List[str]:
    evidence: List[str] = []

    pred_mode = perception.get("pred_mode")
    hazard_type = perception.get("hazard_type")
    view_types = perception.get("view_types")

    if pred_mode or hazard_type:
        evidence.append(
            f"GPA identified input mode as {pred_mode or 'unknown'} "
            f"and hazard type as {hazard_type or 'unknown'}."
        )

    if view_types:
        evidence.append(f"GPA inferred source/view types: {view_types}.")

    if restoration.get("decision"):
        evidence.append(f"GIRA image-quality decision: {restoration['decision']}.")

    restoration_accepted = _normalize_bool(restoration.get("accepted"))
    restoration_needed = _normalize_bool(restoration.get("needs_restoration"))

    if restoration_accepted is True:
        evidence.append("GIRA accepted restored imagery for downstream analysis.")
    elif restoration_accepted is False:
        if restoration_needed is True:
            evidence.append(
                "GIRA detected image-quality degradation and produced restored candidates, "
                "but the restored outputs were rejected because the quality improvement "
                "was insufficient."
            )
        else:
            evidence.append(
                "GIRA did not require restoration; the original imagery remained available "
                "for damage analysis."
            )
    else:
        evidence.append(
            "GIRA reported mixed or unavailable restoration acceptance status for this case."
        )

    if damage.get("analysis_mode"):
        evidence.append(f"GDAA selected damage-analysis mode: {damage['analysis_mode']}.")

    final_severity = infer_final_severity(damage)
    if final_severity:
        evidence.append(f"GDAA estimated final severity as {final_severity}.")

    if damage.get("wildfire_damage_class"):
        evidence.append(
            f"GDAA assigned wildfire building class: {damage['wildfire_damage_class']}."
        )

    building_evidence = _as_list(gdaa_report.get("building_evidence"))
    for item in building_evidence[:3]:
        if isinstance(item, dict):
            part = item.get("building_part", "building element")
            obs = item.get("observation")
            if obs:
                evidence.append(f"Visible evidence on {part}: {obs}")

    rsi_evidence = _as_list(gdaa_report.get("rsi_evidence"))
    for item in rsi_evidence[:2]:
        if isinstance(item, dict):
            obs = item.get("observation")
            if obs:
                evidence.append(f"RSI evidence: {obs}")

    svi_evidence = _as_list(gdaa_report.get("svi_evidence"))
    for item in svi_evidence[:2]:
        if isinstance(item, dict):
            obs = item.get("observation")
            if obs:
                evidence.append(f"SVI evidence: {obs}")

    detections = _as_list(gdaa_report.get("detections"))
    if detections:
        evidence.append(
            f"GDAA localized {len(detections)} visible damage-related or access-related objects."
        )

    return evidence


def build_uncertainties(
    perception: Dict[str, Any],
    restoration: Dict[str, Any],
    damage: Dict[str, Any],
    gdaa_report: Dict[str, Any],
) -> List[str]:
    uncertainties: List[str] = []

    confidence = damage.get("confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.7:
        uncertainties.append("Damage-analysis confidence is below 0.70.")

    mode_confidence = perception.get("mode_confidence")
    if isinstance(mode_confidence, (int, float)) and mode_confidence < 0.7:
        uncertainties.append("Input-mode classification confidence is relatively low.")

    hazard_confidence = perception.get("hazard_confidence")
    if isinstance(hazard_confidence, (int, float)) and hazard_confidence < 0.7:
        uncertainties.append("Hazard-type classification confidence is relatively low.")

    analysis_mode = damage.get("analysis_mode")

    if analysis_mode in ["wildfire_building", "object_detection"]:
        uncertainties.append(
            "Assessment is based on a single image viewpoint unless additional imagery is provided."
        )

    if analysis_mode == "wildfire_building":
        uncertainties.append(
            "Interior damage is not assessed unless directly visible from the image."
        )

    if restoration.get("accepted") is True:
        uncertainties.append(
            "Damage analysis used a restored image; restoration may improve visibility but should not be interpreted as new physical evidence."
        )

    calibration_notes = _as_list(gdaa_report.get("confidence_calibration_notes"))
    for note in calibration_notes:
        if isinstance(note, str) and note.strip():
            uncertainties.append(note.strip())

    return uncertainties


def build_recovery_recommendations(
    hazard_type: str,
    analysis_mode: str,
    damage: Dict[str, Any],
    gdaa_report: Dict[str, Any],
) -> List[str]:
    recommendations: List[str] = []

    hazard = (hazard_type or "").lower()

    if analysis_mode == "wildfire_building":
        recommendations.extend(
            [
                "Conduct field verification of the exterior and interior building condition.",
                "Inspect windows, doors, façade, roofline, and utility connections for fire-related damage.",
                "Confirm whether the structure is safe for entry before repair or reconstruction decisions.",
            ]
        )

    elif analysis_mode == "cross_view_svi_rsi":
        recommendations.extend(
            [
                "Use the satellite/aerial view to prioritize affected zones and the street-view evidence to verify local accessibility.",
                "Inspect roads, vegetation debris, localized flooding, and visible structural elements on the ground.",
            ]
        )

    elif analysis_mode == "pre_post_svi":
        recommendations.extend(
            [
                "Verify the detected pre/post structural changes through field inspection or additional imagery.",
                "Prioritize assets showing disappearance, collapse, severe alteration, or access obstruction.",
            ]
        )

    elif analysis_mode == "object_detection":
        recommendations.extend(
            [
                "Use detected bounding boxes as approximate spatial cues for field verification.",
                "Prioritize road obstructions, damaged structures, and critical infrastructure detections.",
            ]
        )

    if "hurricane" in hazard:
        recommendations.append(
            "Check for secondary hazards such as standing water, debris accumulation, and compromised access routes."
        )

    if "wildfire" in hazard:
        recommendations.append(
            "Check for residual heat, smoke exposure, façade damage, utility hazards, and hidden interior fire damage."
        )

    if not recommendations:
        recommendations.append(
            "Perform field verification and compare with additional imagery before operational decisions."
        )

    return recommendations

def build_final_report_text(
    case_id: str,
    perception: Dict[str, Any],
    restoration: Dict[str, Any],
    damage: Dict[str, Any],
    evidence_chain: List[str],
    uncertainties: List[str],
    recovery_recommendations: List[str],
    gdaa_report: Dict[str, Any],
) -> str:
    """
    Build a richer technical final report for the HERMES GRA output.
    The text is deterministic and grounded in GPA, GIRA, and GDAA outputs.
    """

    hazard_type = (
        perception.get("hazard_type")
        or damage.get("hazard_type")
        or gdaa_report.get("hazard_type")
        or "unknown"
    )

    input_mode = perception.get("pred_mode") or "unknown"

    mode_description = {
        "post_sv_single": "a single post-event street-view image",
        "post_rs_single": "a single post-event remote-sensing image",
        "post_paired_sv_rs": "a paired post-event street-view and remote-sensing case",
        "bitemporal_sv": "a before/after street-view comparison",
        "bitemporal_rs": "a before/after remote-sensing comparison",
    }.get(input_mode, input_mode)

    analysis_mode = damage.get("analysis_mode") or gdaa_report.get("analysis_mode") or "unknown"

    severity = (
        damage.get("severity")
        or damage.get("three_level_severity")
        or gdaa_report.get("three_level_severity")
        or "unknown"
    )

    overall_severity = (
        damage.get("overall_severity")
        or gdaa_report.get("overall_severity")
        or "unknown"
    )

    three_level_severity = (
        damage.get("three_level_severity")
        or gdaa_report.get("three_level_severity")
        or severity
    )

    domain_class = (
        damage.get("domain_specific_class")
        or damage.get("wildfire_damage_class")
        or gdaa_report.get("domain_specific_class")
        or gdaa_report.get("wildfire_damage_class")
        or "not applicable"
    )

    damage_percent_range = (
        damage.get("damage_percent_range")
        or gdaa_report.get("damage_percent_range")
        or "not specified"
    )

    confidence = damage.get("confidence")
    if confidence is None:
        confidence = gdaa_report.get("confidence")

    confidence_text = "not reported" if confidence is None else str(confidence)

    final_path = (
        restoration.get("final_path")
        or restoration.get("final_paths")
        or "not available"
    )

    q_before = restoration.get("Q_before")
    q_after = restoration.get("Q_after")
    delta_q = restoration.get("delta_Q")
    restoration_decision = restoration.get("decision") or "No restoration decision available."

    restoration_needed = _normalize_bool(restoration.get("needed"))
    restoration_accepted = _normalize_bool(restoration.get("accepted"))

    if restoration_needed and restoration_accepted:
        restoration_status_text = (
            "The GeoImageRestoration Agent determined that restoration was required "
            "and accepted the restored image as the downstream analytical input."
        )
    elif restoration_needed and not restoration_accepted:
        restoration_status_text = (
            "The GeoImageRestoration Agent detected potential image-quality degradation, "
            "but the restored candidate was not accepted because it did not provide "
            "sufficient quality improvement according to the configured quality criteria."
        )
    else:
        restoration_status_text = (
            "The GeoImageRestoration Agent determined that restoration was not required, "
            "therefore the original image was retained as the final input for damage analysis."
        )

    quality_score_text = []
    if q_before is not None:
        quality_score_text.append(f"Q_before={q_before}")
    if q_after is not None:
        quality_score_text.append(f"Q_after={q_after}")
    if delta_q is not None:
        quality_score_text.append(f"delta_Q={delta_q}")

    if quality_score_text:
        quality_text = ", ".join(quality_score_text)
    else:
        quality_text = "no image-quality scores were available"

    affected_assets = _as_list(gdaa_report.get("affected_assets"))
    affected_assets_text = (
        ", ".join(str(asset) for asset in affected_assets)
        if affected_assets
        else "not explicitly listed"
    )

    building_evidence = _as_list(gdaa_report.get("building_evidence"))
    evidence_sentences: List[str] = []

    for item in building_evidence[:4]:
        if isinstance(item, dict):
            part = str(item.get("building_part") or "observed building component").strip()
            observation = str(item.get("observation") or "").strip().rstrip(".")
            interpretation = str(item.get("damage_interpretation") or "").strip().rstrip(".")

            if observation and interpretation:
                evidence_sentences.append(
                    f"{part}: {observation}. Interpretation: {interpretation}."
                )
            elif observation:
                evidence_sentences.append(f"{part}: {observation}.")
            elif interpretation:
                evidence_sentences.append(f"{part}: {interpretation}.")

    if not evidence_sentences:
        evidence_sentences = [str(e) for e in evidence_chain[-4:]]

    detailed_evidence_text = " ".join(evidence_sentences)

    if not detailed_evidence_text:
        detailed_evidence_text = (
            "No detailed object-level evidence was available beyond the structured "
            "damage-classification output."
        )

    uncertainty_text = " ".join(str(u) for u in uncertainties)
    if not uncertainty_text:
        uncertainty_text = (
            "No explicit uncertainty statement was generated; however, the assessment "
            "should still be treated as image-based and subject to validation."
        )

    recommendation_text = " ".join(str(r) for r in recovery_recommendations)
    if not recommendation_text:
        recommendation_text = (
            "Field verification is recommended before operational, engineering, "
            "or reconstruction decisions are made."
        )

    paragraph_1 = (
        f"HERMES processed case {case_id} as a {hazard_type} damage-assessment "
        f"scenario using {mode_description}. The GeoPerception Agent classified the "
        f"input configuration as {input_mode} and associated the scene with the "
        f"{hazard_type} hazard context. This perception output provided the routing "
        f"basis for the subsequent image-quality assessment, damage-analysis, and "
        f"final reasoning stages."
    )

    paragraph_2 = (
        f"During the image-quality stage, {restoration_status_text} "
        f"The restoration decision was: {restoration_decision.strip()} "
        f"The final image used for damage analysis was {final_path}. "
        f"The available image-quality indicators were {quality_text}."
    )

    paragraph_3 = (
        f"The GeoDamageAnalysis Agent selected the {analysis_mode} workflow. "
        f"For this wildfire-building assessment, the domain-specific damage class "
        f"was {domain_class}, corresponding to an estimated visible damage range "
        f"of {damage_percent_range}. The domain-specific output reported overall "
        f"severity as {overall_severity}, while the normalized HERMES three-level "
        f"severity taxonomy maps the case to {three_level_severity}. This distinction "
        f"reflects the harmonization between a wildfire-specific percentage-based "
        f"damage scale and the system-wide HERMES severity representation. The final "
        f"GRA severity is therefore {severity}, with reported confidence "
        f"{confidence_text}."
    )

    paragraph_4 = (
        f"The main affected assets identified in the scene are: {affected_assets_text}. "
        f"The evidence supporting the assessment includes: {detailed_evidence_text} "
        f"Taken together, these observations indicate visible exterior wildfire impact, "
        f"primarily affecting non-structural or exposed façade components such as "
        f"windows, doors, and exterior wall surfaces. No direct evidence of complete "
        f"structural collapse is available from the current image viewpoint."
    )

    paragraph_5 = (
        f"This assessment should be interpreted as an image-based rapid "
        f"situational-awareness product rather than a definitive structural or "
        f"engineering inspection. The main limitations are as follows: {uncertainty_text} "
        f"Accordingly, HERMES output should support prioritization and triage, but "
        f"operational decisions should be validated through field inspection or "
        f"additional imagery where available. Recommended follow-up actions include: "
        f"{recommendation_text}"
    )

    return "\n\n".join(
        [
            paragraph_1,
            paragraph_2,
            paragraph_3,
            paragraph_4,
            paragraph_5,
        ]
    )

def build_report_images(
    perception: Dict[str, Any],
    restoration: Dict[str, Any],
    damage: Dict[str, Any],
    gdaa_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build scenario-aware image list for the HTML report.
    """

    analysis_mode = (
        damage.get("analysis_mode")
        or gdaa_report.get("analysis_mode")
        or "unknown"
    )

    images: List[Dict[str, Any]] = []

    if analysis_mode == "wildfire_building":
        path = (
            restoration.get("final_path")
            or gdaa_report.get("image_path")
            or gdaa_report.get("input_path")
        )

        if path:
            images.append(
                {
                    "role": "post_event",
                    "label": "Post-event Street-View Image",
                    "path": path,
                }
            )

    elif analysis_mode == "pre_post_svi":
        pre_path = gdaa_report.get("pre_image_path")
        post_path = gdaa_report.get("post_image_path")

        if pre_path:
            images.append(
                {
                    "role": "pre_event",
                    "label": "Pre-event Street-View Image",
                    "path": pre_path,
                }
            )

        if post_path:
            images.append(
                {
                    "role": "post_event",
                    "label": "Post-event Street-View Image",
                    "path": post_path,
                }
            )

    elif analysis_mode == "cross_view_svi_rsi":
        rs_path = (
            gdaa_report.get("rs_image_path")
            or gdaa_report.get("rsi_image_path")
            or gdaa_report.get("satellite_image_path")
            or gdaa_report.get("remote_sensing_image_path")
        )

        sv_path = (
            gdaa_report.get("sv_image_path")
            or gdaa_report.get("svi_image_path")
            or gdaa_report.get("street_view_image_path")
        )

        if rs_path:
            images.append(
                {
                    "role": "remote_sensing",
                    "label": "Remote-Sensing / Satellite Image",
                    "path": rs_path,
                }
            )

        if sv_path:
            images.append(
                {
                    "role": "street_view",
                    "label": "Street-View Image",
                    "path": sv_path,
                }
            )

    elif analysis_mode == "object_detection":
        input_path = (
            gdaa_report.get("image_path")
            or gdaa_report.get("input_path")
            or restoration.get("final_path")
        )

        annotated_path = (
            gdaa_report.get("annotated_image_path")
            or gdaa_report.get("visualization_path")
            or gdaa_report.get("output_image_path")
        )

        if input_path:
            images.append(
                {
                    "role": "input",
                    "label": "Input Image",
                    "path": input_path,
                }
            )

        if annotated_path:
            images.append(
                {
                    "role": "annotated_output",
                    "label": "Annotated Detection Output",
                    "path": annotated_path,
                }
            )

    if not images:
        final_paths = restoration.get("final_paths")
        final_path = restoration.get("final_path")

        if isinstance(final_paths, list):
            for index, path in enumerate(final_paths, start=1):
                if path:
                    images.append(
                        {
                            "role": f"image_{index}",
                            "label": f"Analysis Image {index}",
                            "path": path,
                        }
                    )

        elif final_path:
            images.append(
                {
                    "role": "analysis_image",
                    "label": "Analysis Image",
                    "path": final_path,
                }
            )

    return images


def build_deterministic_report(
    gpa_report: Dict[str, Any],
    gira_report: Dict[str, Any],
    gdaa_report: Dict[str, Any],
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    perception = extract_perception_summary(gpa_report)
    restoration = extract_restoration_summary(gira_report)
    damage = extract_damage_summary(gdaa_report)

    resolved_case_id = (
        case_id
        or damage.get("pair_id")
        or gdaa_report.get("pair_id")
        or "unknown_case"
    )

    hazard_type = damage.get("hazard_type") or perception.get("hazard_type") or "unknown"
    input_mode = perception.get("pred_mode") or "unknown"
    analysis_mode = damage.get("analysis_mode") or "unknown"
    final_severity = infer_final_severity(damage)

    rs_image_path = (
        damage.get("rs_image_path")
        or _path_for_source_type(
            restoration.get("final_paths") or restoration.get("input_paths"),
            restoration.get("source_types"),
            "RS",
        )
    )

    sv_image_path = (
        damage.get("sv_image_path")
        or _path_for_source_type(
            restoration.get("final_paths") or restoration.get("input_paths"),
            restoration.get("source_types"),
            "SV",
        )
    )

    evidence_chain = build_evidence_chain(perception, restoration, damage, gdaa_report)
    uncertainties = build_uncertainties(perception, restoration, damage, gdaa_report)
    recommendations = build_recovery_recommendations(
        hazard_type=hazard_type,
        analysis_mode=analysis_mode,
        damage=damage,
        gdaa_report=gdaa_report,
    )

    object_summary = build_object_recognition_summary(gdaa_report)

    severity_rationale = (
        gdaa_report.get("reasoning")
        or damage.get("reasoning")
        or "Severity rationale was derived from the available structured damage-analysis indicators."
    )

    causal_interpretation = (
        f"The observed damage pattern is interpreted as {hazard_type}-related "
        f"based on the perception output and the damage-analysis evidence."
    )


    final_report = build_final_report_text(
        case_id=resolved_case_id,
        perception=perception,
        restoration=restoration,
        damage=damage,
        evidence_chain=evidence_chain,
        uncertainties=uncertainties,
        recovery_recommendations=recommendations,
        gdaa_report=gdaa_report,
    )



    return {
        "agent": "GeoReasoningAgent",
        "case_id": resolved_case_id,
        "type_of_disaster": hazard_type,
        "input_mode": input_mode,
        "image_restoration": {
            "needed": restoration.get("needs_restoration"),
            "accepted": restoration.get("accepted"),
            "final_path": restoration.get("final_path"),
            "final_paths": restoration.get("final_paths"),
            "restored_path": restoration.get("restored_path"),
            "restored_paths": restoration.get("restored_paths"),
            "Q_before": restoration.get("Q_before"),
            "Q_after": restoration.get("Q_after"),
            "delta_Q": restoration.get("delta_Q"),
            "decision": restoration.get("decision"),
            "results": restoration.get("results"),
            "num_images": restoration.get("num_images"),
            "mode": restoration.get("mode"),
            "source_types": restoration.get("source_types"),
            "input_paths": restoration.get("input_paths"),
        },
        "damage_assessment": {
            "analysis_mode": damage.get("analysis_mode"),
            "damage_presence": damage.get("damage_presence"),
            "severity": final_severity,
            "overall_severity": damage.get("overall_severity"),
            "three_level_severity": (
                damage.get("three_level_severity")
                or damage.get("overall_severity")
            ),
            "domain_specific_class": (
                damage.get("domain_specific_class")
                or damage.get("wildfire_damage_class")
            ),
            "wildfire_damage_class": damage.get("wildfire_damage_class"),
            "damage_percent_range": damage.get("damage_percent_range"),
            "confidence": damage.get("confidence"),

            # Scenario-specific structured fields
            "affected_assets": damage.get("affected_assets"),
            "objects_detected": damage.get("objects_detected"),
            "change_evidence": damage.get("change_evidence"),
            "reasoning": damage.get("reasoning"),
            "confidence_calibration_notes": damage.get("confidence_calibration_notes"),
            "pre_image_path": damage.get("pre_image_path"),
            "post_image_path": damage.get("post_image_path"),
            "rs_image_path": rs_image_path,
            "sv_image_path": sv_image_path,
            "image_path": damage.get("image_path"),
            "detections": damage.get("detections"),
            "num_detections": damage.get("num_detections"),
            "object_counts": damage.get("object_counts"),
            "category_counts": damage.get("category_counts"),
            "spatial_summary": damage.get("spatial_summary"),
            "annotated_image_path": damage.get("annotated_image_path"),
        },
        "report_images": build_report_images(
            perception=perception,
            restoration=restoration,
            damage=damage,
            
            gdaa_report=gdaa_report,
        ),
        "object_recognition_summary": object_summary,
        "evidence_chain": evidence_chain,
        "severity_rationale": severity_rationale,
        "causal_interpretation": causal_interpretation,
        "uncertainties": uncertainties,
        "recovery_recommendations": recommendations,
        "final_report": final_report,
        "machine_readable_summary": {
            "hazard_type": hazard_type,
            "pred_mode": input_mode,
            "analysis_mode": analysis_mode,
            "final_path": restoration.get("final_path"),
            "final_paths": restoration.get("final_paths"),
            "annotated_image_path": damage.get("annotated_image_path"),
            "rs_image_path": rs_image_path,
            "sv_image_path": sv_image_path,
            "num_detections": damage.get("num_detections"),
            "object_counts": damage.get("object_counts"),
            "category_counts": damage.get("category_counts"),
            "detections": damage.get("detections"),
            "spatial_summary": damage.get("spatial_summary"),
        },
        "llm_used": False,
        "reasoning_model": None,
    }


def build_llm_prompt(
    deterministic_report: Dict[str, Any],
    gpa_report: Dict[str, Any],
    gira_report: Dict[str, Any],
    gdaa_report: Dict[str, Any],
) -> str:
    """
    Build a scenario-aware LLM prompt for the GeoReasoning Agent.

    The LLM is allowed to choose the most appropriate report form and improve
    the narrative explanation. It is not allowed to modify locked analytical
    facts such as hazard type, analysis mode, severity, confidence, or paths.
    """

    damage = deterministic_report.get("damage_assessment", {}) or {}

    locked_facts = {
        "case_id": deterministic_report.get("case_id"),
        "type_of_disaster": deterministic_report.get("type_of_disaster"),
        "input_mode": deterministic_report.get("input_mode"),
        "analysis_mode": damage.get("analysis_mode"),
        "damage_presence": damage.get("damage_presence"),
        "severity": damage.get("severity"),
        "overall_severity": damage.get("overall_severity"),
        "three_level_severity": damage.get("three_level_severity"),
        "domain_specific_class": damage.get("domain_specific_class"),
        "damage_percent_range": damage.get("damage_percent_range"),
        "confidence": damage.get("confidence"),
        "image_restoration": deterministic_report.get("image_restoration"),
        "machine_readable_summary": deterministic_report.get("machine_readable_summary"),
        "report_images": deterministic_report.get("report_images"),
    }

    payload = {
        "locked_facts": locked_facts,
        "deterministic_report": deterministic_report,
        "gpa_report": gpa_report,
        "gira_report": gira_report,
        "gdaa_report": gdaa_report,
    }

    return f"""
You are the GeoReasoning Agent (GRA) of HERMES, a multi-agent geospatial disaster assessment system.

Your role is to synthesize structured outputs from:
- GPA: GeoPerception Agent
- GIRA: GeoImageRestoration Agent
- GDAA: GeoDamageAnalysis Agent

You must choose the correct report form based on the scenario. Use the fields:
- analysis_mode
- input_mode / pred_mode
- hazard_type
- source_type / source_types
- evidence fields available in GDAA

Supported report forms:

1. wildfire_building
Use this when analysis_mode is "wildfire_building".
The report should discuss wildfire-specific building damage class, visible exterior fire impact, damaged assets such as windows, doors, façade, exterior walls, and percentage-based wildfire damage class.

2. pre_post_svi
Use this when analysis_mode is "pre_post_svi".
The report should discuss before/after street-view comparison, temporal change evidence, disappearance of structures, structural change, vegetation change, road/access condition, debris, and confidence limitations from bitemporal panoramic imagery.
Do not use wildfire-building wording unless the hazard is wildfire and the GDAA evidence explicitly supports it.

3. cross_view_svi_rsi
Use this when analysis_mode is "cross_view_svi_rsi".
The report should discuss cross-view corroboration between satellite/remote-sensing imagery and street-view imagery, modality agreement or disagreement, asset-level consistency, and uncertainty due to different viewpoints/scales.

4. object_detection
Use this when analysis_mode is "object_detection".
The report should discuss detected objects, damage indicators, localization, bounding-box or annotated-image evidence if available, and limitations of single-image object detection.

5. generic_damage_report
Use this only if the scenario cannot be confidently mapped to one of the above.

Hard constraints:
- Do NOT change the locked facts.
- Do NOT invent objects, damage classes, image paths, confidence values, or severity values.
- If a field is missing, say that it was not available.
- The final report must be scenario-aware.
- The final report must avoid wildfire façade/windows language unless analysis_mode is wildfire_building.
- The final report must be written as a technical situational-awareness report, not as a casual explanation.
- Use the GDAA evidence as the primary basis for severity rationale.
- Use GPA and GIRA only for routing, hazard context, input mode, restoration status, and image-quality context.

Scenario-specific reporting rules:
- If report_template_selected or analysis_mode is "pre_post_svi", do NOT state that the domain-specific damage class is unavailable. Instead, summarize the before/after change evidence, affected assets, and active damage indicators.
- If report_template_selected or analysis_mode is "cross_view_svi_rsi", do NOT state that the domain-specific damage class is unavailable. Instead, summarize cross-view agreement, affected assets, and active damage indicators.
- Only discuss "domain-specific damage class" or "damage percent range" for wildfire_building reports.
- For pre_post_svi reports, the absence of a wildfire-style class is not a missing field; it is not applicable to that report form.
- For pre_post_svi reports, final_report must focus on temporal evidence, affected assets, damage indicators, severity rationale, restoration status, uncertainty, and verification needs.
- For pre_post_svi reports, do NOT state that the three-level severity classification is unavailable when severity or overall_severity is available. Use the reported severity / overall_severity as the normalized HERMES severity.
- For pre_post_svi reports, do NOT use the phrases "domain-specific damage class", "damage class", "damage percent range", "not applicable", or "not available" anywhere in executive_summary, severity_rationale, causal_interpretation, or final_report.
- For pre_post_svi reports, replace any discussion of damage class or damage percent range with a concise description of active damage indicators, affected assets, and temporal change evidence.
- For cross_view_svi_rsi reports, do NOT use the phrases "wildfire-building assessment", "wildfire building", "domain-specific damage class", "damage class", "damage percent range", "not applicable", or "not available" anywhere in executive_summary, severity_rationale, causal_interpretation, or final_report.
- For cross_view_svi_rsi reports, replace any discussion of damage class or damage percent range with cross-view evidence, modality agreement/disagreement, affected assets, and active damage indicators.
- For cross_view_svi_rsi reports, do NOT describe the case as a pre/post or before/after comparison. It is a paired post-event cross-view assessment using remote-sensing and street-view imagery.
- For cross_view_svi_rsi reports, explicitly distinguish RSI evidence from SVI evidence when such fields are available.
- For cross_view_svi_rsi reports, restoration wording must follow image_restoration exactly. If restoration has mixed accepted values, state that restoration was mixed across the image pair and identify which image was accepted or rejected if available. Do NOT say restoration was not required unless image_restoration.needed is false for all images.
- For cross_view_svi_rsi reports, do NOT use the phrases "wildfire-building assessment", "wildfire building", "domain-specific damage class", "damage class", "damage percent range", "percentage-based damage scale", "percentage-based damage", "not applicable", or "not available" anywhere in executive_summary, severity_rationale, causal_interpretation, or final_report.

Return STRICT JSON only with this schema:

{{
  "report_template_selected": "wildfire_building | pre_post_svi | cross_view_svi_rsi | object_detection | generic_damage_report",
  "scenario_interpretation": "brief explanation of why this template was selected",
  "executive_summary": "one concise technical paragraph",
  "object_recognition_summary": "short summary of detected objects, changed assets, or evidence indicators",
  "evidence_chain": [
    "evidence item 1",
    "evidence item 2"
  ],
  "severity_rationale": "technical explanation of why the reported severity follows from the evidence",
  "causal_interpretation": "hazard-specific interpretation of the observed damage pattern",
  "uncertainties": [
    "uncertainty or limitation 1",
    "uncertainty or limitation 2"
  ],
  "recovery_recommendations": [
    "recommendation 1",
    "recommendation 2"
  ],
  "final_report": "multi-paragraph technical final report"
}}

Input data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def call_gemini_reasoning(prompt: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    selected_model = model_name or GEMINI_REASONING_MODEL

    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    return json.loads(response.text)


def _active_indicator_names(objects_detected: Any) -> List[str]:
    if not isinstance(objects_detected, dict):
        return []

    active = []

    for key, value in objects_detected.items():
        try:
            if float(value) > 0:
                active.append(str(key))
        except Exception:
            if value:
                active.append(str(key))

    return active


def _join_text_items(value: Any, fallback: str = "not specified") -> str:
    if value is None:
        return fallback

    if isinstance(value, str):
        return value if value.strip() else fallback

    if isinstance(value, dict):
        items = [
            f"{key}: {val}"
            for key, val in value.items()
            if val is not None and str(val).strip()
        ]
        return "; ".join(items) if items else fallback

    if isinstance(value, (list, tuple)):
        items = [
            str(item)
            for item in value
            if item is not None and str(item).strip()
        ]
        return "; ".join(items) if items else fallback

    return str(value)


def _fmt_q(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "not reported"


def _restoration_sentence_from_structured(restoration: Dict[str, Any]) -> str:
    results = restoration.get("results")

    if isinstance(results, list) and results:
        sentences = []

        for item in results:
            if not isinstance(item, dict):
                continue

            source_type = item.get("source_type") or "image"
            image_name = item.get("image") or "unnamed image"
            accepted = item.get("accepted")
            needed = item.get("needs_restoration")
            q_before = _fmt_q(item.get("Q_before"))
            q_after = _fmt_q(item.get("Q_after"))
            final_path = item.get("final_path") or "not reported"

            if needed is False:
                sentences.append(
                    f"Restoration was not required for the {source_type} image "
                    f"({image_name}); the final path was {final_path}."
                )
            elif accepted is True:
                sentences.append(
                    f"Restoration was accepted for the {source_type} image "
                    f"({image_name}); Q improved from {q_before} to {q_after}, "
                    f"and the final path was {final_path}."
                )
            elif accepted is False:
                sentences.append(
                    f"Restoration was rejected for the {source_type} image "
                    f"({image_name}); Q changed from {q_before} to {q_after}, "
                    f"which was insufficient for acceptance, and the retained "
                    f"final path was {final_path}."
                )

        if sentences:
            return " ".join(sentences)

    accepted = restoration.get("accepted")

    needed = restoration.get("needed")
    if needed is None:
        needed = restoration.get("needs_restoration")

    final_path = restoration.get("final_path") or "not reported"
    restored_path = restoration.get("restored_path") or "not reported"
    q_before = _fmt_q(restoration.get("Q_before"))
    q_after = _fmt_q(restoration.get("Q_after"))

    if needed is True and accepted is False:
        return (
            f"restoration was attempted but rejected because the Q improvement "
            f"was insufficient. Q changed from {q_before} to {q_after}. "
            f"The retained final image path was {final_path}; the rejected restored "
            f"candidate was {restored_path}."
        )

    if needed is True and accepted is True:
        return (
            f"restoration was accepted. Q changed from {q_before} to {q_after}. "
            f"The final restored image path was {final_path}."
        )

    if needed is False:
        return (
            f"restoration was not required. The retained final image path was "
            f"{final_path}."
        )

    if isinstance(needed, list) and all(item is False for item in needed):
        return "Restoration was not required for the image set."

    if isinstance(accepted, list):
        accepted_count = sum(1 for item in accepted if item is True)
        total = len(accepted)
        return (
            f"Restoration was partially accepted across the image set "
            f"({accepted_count}/{total} accepted)."
        )

    return _join_text_items(
        restoration.get("decision"),
        fallback="Restoration status was not reported.",
    )


def _safe_non_wildfire_final_report(report: Dict[str, Any]) -> Optional[str]:
    damage = report.get("damage_assessment", {}) or {}
    restoration = report.get("image_restoration", {}) or {}

    scenario = damage.get("analysis_mode")
    case_id = report.get("case_id") or "unknown_case"
    hazard_type = report.get("type_of_disaster") or "unknown"
    input_mode = report.get("input_mode") or "unknown"

    severity = (
        damage.get("severity")
        or damage.get("three_level_severity")
        or damage.get("overall_severity")
        or "unknown"
    )

    confidence = damage.get("confidence")
    confidence_text = "not reported" if confidence is None else str(confidence)

    affected_assets = _join_text_items(
        damage.get("affected_assets"),
        fallback="no structured affected assets reported",
    )

    active_indicators = _join_text_items(
        _active_indicator_names(damage.get("objects_detected")),
        fallback="no active structured damage indicators reported",
    )

    restoration_text = _restoration_sentence_from_structured(restoration)

    uncertainties = _join_text_items(
        report.get("uncertainties"),
        fallback="No additional structured uncertainty statement was reported.",
    )

    recommendations = _join_text_items(
        report.get("recovery_recommendations"),
        fallback="No structured recovery recommendation was reported.",
    )

    evidence_chain = report.get("evidence_chain") or []

    if scenario == "cross_view_svi_rsi":
        rsi_evidence = [
            str(item)
            for item in evidence_chain
            if "RSI evidence" in str(item)
            or "remote-sensing" in str(item).lower()
            or "satellite" in str(item).lower()
        ]

        svi_evidence = [
            str(item)
            for item in evidence_chain
            if "SVI evidence" in str(item)
            or "street-view" in str(item).lower()
        ]

        rsi_text = _join_text_items(
            rsi_evidence,
            fallback="Remote-sensing evidence was used for area-level damage context.",
        )

        svi_text = _join_text_items(
            svi_evidence,
            fallback="Street-view evidence was used for ground-level damage confirmation.",
        )

        return (
            f"HERMES processed case {case_id} as a {hazard_type} damage-assessment "
            f"scenario using paired post-event remote-sensing and street-view imagery. "
            f"The input mode was {input_mode}, and the GeoDamageAnalysis Agent selected "
            f"the {scenario} workflow. The final normalized HERMES severity is "
            f"{severity}, with reported confidence {confidence_text}.\n\n"
            f"During the image-quality stage, {restoration_text}\n\n"
            f"The cross-view assessment identified affected assets including "
            f"{affected_assets}. Active damage indicators include {active_indicators}. "
            f"Remote-sensing evidence: {rsi_text} Street-view evidence: {svi_text}\n\n"
            f"The severity assessment is based on consistency between area-level "
            f"remote-sensing observations and ground-level street-view evidence. "
            f"Remaining uncertainty is driven by viewpoint, scale, and visibility "
            f"differences between the two modalities. Limitations: {uncertainties}\n\n"
            f"Recommended response actions: {recommendations}"
        )

    if scenario == "pre_post_svi":
        change_evidence = _join_text_items(
            damage.get("change_evidence"),
            fallback="temporal street-view change evidence was used for damage assessment",
        )

        return (
            f"HERMES processed case {case_id} as a {hazard_type} damage-assessment "
            f"scenario using before/after street-view imagery. The input mode was "
            f"{input_mode}, and the GeoDamageAnalysis Agent selected the {scenario} "
            f"workflow. The final normalized HERMES severity is {severity}, with "
            f"reported confidence {confidence_text}.\n\n"
            f"During the image-quality stage, {restoration_text}\n\n"
            f"The temporal assessment identified affected assets including "
            f"{affected_assets}. Active damage indicators include {active_indicators}. "
            f"The main change evidence is: {change_evidence}\n\n"
            f"The severity assessment is based on visible temporal change between the "
            f"pre-disaster and post-disaster street-view observations. Limitations: "
            f"{uncertainties}\n\n"
            f"Recommended response actions: {recommendations}"
        )

    if scenario == "object_detection":
        object_counts = damage.get("object_counts") or damage.get("objects_detected") or {}
        category_counts = damage.get("category_counts") or {}
        num_detections = damage.get("num_detections")
        spatial_summary = damage.get("spatial_summary") or "Object detections were used as spatial evidence."
        annotated_image_path = (
            damage.get("annotated_image_path")
            or (report.get("machine_readable_summary", {}) or {}).get("annotated_image_path")
        )

        objects_text = _join_text_items(
            object_counts,
            fallback="no structured object counts reported",
        )

        categories_text = _join_text_items(
            category_counts,
            fallback="no structured detection categories reported",
        )

        detection_count_text = (
            str(num_detections)
            if num_detections is not None
            else "an unspecified number of"
        )

        return (
            f"HERMES processed case {case_id} as a {hazard_type} damage-assessment "
            f"scenario using a single post-event street-view image. The input mode "
            f"was {input_mode}, and the GeoDamageAnalysis Agent selected the "
            f"{scenario} workflow.\n\n"
            f"During the image-quality stage, {restoration_text}\n\n"
            f"The object-detection analysis localized {detection_count_text} "
            f"damage-related or access-related objects. The structured object counts "
            f"were: {objects_text}. The detection categories were: {categories_text}. "
            f"{spatial_summary} The annotated detection output was saved at "
            f"{annotated_image_path}.\n\n"
            f"This report should be interpreted as a rapid situational-awareness "
            f"product based on localized detections rather than a complete structural "
            f"engineering assessment. The detections provide spatial cues for field "
            f"verification of damaged structures, road obstructions, access routes, "
            f"and debris locations.\n\n"
            f"Recommended response actions: {recommendations}"
        )

    return None


def _final_report_violates_scenario_rules(report: Dict[str, Any]) -> bool:
    damage = report.get("damage_assessment", {}) or {}
    restoration = report.get("image_restoration", {}) or {}

    scenario = damage.get("analysis_mode")
    final_report = str(report.get("final_report") or "").lower()

    if scenario == "cross_view_svi_rsi":
        forbidden_phrases = [
            "wildfire-building",
            "wildfire building",
            "domain-specific damage class",
            "damage class",
            "damage percent range",
            "percentage-based damage scale",
            "percentage-based damage",
            "percentage-based",
            "not applicable",
            "not available",
            "before/after comparison",
            "pre-event",
            "before-event",
        ]

        if any(phrase in final_report for phrase in forbidden_phrases):
            return True

        needed = restoration.get("needed") or restoration.get("needs_restoration")

        if (
            "restoration was not required" in final_report
            and not (
                isinstance(needed, list)
                and needed
                and all(item is False for item in needed)
            )
        ):
            return True

    if scenario == "pre_post_svi":
        forbidden_phrases = [
            "wildfire-building",
            "wildfire building",
            "domain-specific damage class",
            "damage class",
            "damage percent range",
            "not applicable",
            "not available",
            "three-level severity classification was not available",
        ]

        if any(phrase in final_report for phrase in forbidden_phrases):
            return True

    if scenario == "object_detection":
        forbidden_phrases = [
            "wildfire-building",
            "wildfire building",
            "domain-specific damage class",
            "damage percent range",
            "percentage-based damage",
            "not applicable",
        ]

        if any(phrase in final_report for phrase in forbidden_phrases):
            return True

        needed = restoration.get("needed")
        if needed is None:
            needed = restoration.get("needs_restoration")

        accepted = restoration.get("accepted")

        if (
            "restoration was not required" in final_report
            and (needed is True or accepted is False)
        ):
            return True

    return False


def merge_llm_with_deterministic(
    llm_report: Dict[str, Any],
    deterministic_report: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """
    Preserve key deterministic fields from the pipeline while allowing the LLM
    to improve narrative fields.
    """
    if not isinstance(llm_report, dict):
        return deterministic_report

    merged = dict(deterministic_report)
    merged.update(llm_report)

    # Preserve high-trust machine-generated fields.
    merged["agent"] = "GeoReasoningAgent"
    merged["case_id"] = deterministic_report.get("case_id")
    merged["type_of_disaster"] = deterministic_report.get("type_of_disaster")
    merged["input_mode"] = deterministic_report.get("input_mode")
    merged["image_restoration"] = deterministic_report.get("image_restoration")
    merged["damage_assessment"] = deterministic_report.get("damage_assessment")
    merged["machine_readable_summary"] = deterministic_report.get("machine_readable_summary")
    merged["report_images"] = deterministic_report.get("report_images", [])

    merged["llm_used"] = True
    merged["reasoning_model"] = model_name

    # validate final_report
    if _final_report_violates_scenario_rules(merged):
        safe_final_report = _safe_non_wildfire_final_report(merged)
        if safe_final_report:
            merged["final_report"] = safe_final_report
            merged["final_report_guard_applied"] = True
    else:
        merged["final_report_guard_applied"] = False

    return merged


def report_to_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    case_id = report.get("case_id", "unknown_case")

    lines.append(f"# HERMES Situational Awareness Report")
    lines.append("")
    lines.append(f"**Case ID:** {case_id}")
    lines.append(f"**Type of disaster:** {report.get('type_of_disaster', 'unknown')}")
    lines.append(f"**Input mode:** {report.get('input_mode', 'unknown')}")
    lines.append("")

    damage = report.get("damage_assessment", {})
    lines.append("## Final Damage Assessment")
    lines.append("")
    lines.append(f"- **Analysis mode:** {damage.get('analysis_mode')}")
    lines.append(f"- **Damage presence:** {damage.get('damage_presence')}")
    lines.append(f"- **Severity:** {damage.get('severity')}")
    lines.append(f"- **Domain-specific class:** {damage.get('domain_specific_class')}")
    lines.append(f"- **Damage percent range:** {damage.get('damage_percent_range')}")
    lines.append(f"- **Confidence:** {damage.get('confidence')}")
    lines.append("")

    restoration = report.get("image_restoration", {})
    lines.append("## Image Restoration")
    lines.append("")
    lines.append(f"- **Needed:** {restoration.get('needed')}")
    lines.append(f"- **Accepted:** {restoration.get('accepted')}")
    lines.append(f"- **Final path:** `{restoration.get('final_path')}`")
    lines.append(f"- **Decision:** {restoration.get('decision')}")
    lines.append("")

    lines.append("## Object Recognition Summary")
    lines.append("")
    lines.append(str(report.get("object_recognition_summary", "")))
    lines.append("")

    lines.append("## Evidence Chain")
    lines.append("")
    for item in _as_list(report.get("evidence_chain")):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Severity Rationale")
    lines.append("")
    lines.append(str(report.get("severity_rationale", "")))
    lines.append("")

    lines.append("## Causal Interpretation")
    lines.append("")
    lines.append(str(report.get("causal_interpretation", "")))
    lines.append("")

    lines.append("## Uncertainties and Limitations")
    lines.append("")
    for item in _as_list(report.get("uncertainties")):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Recovery / Verification Recommendations")
    lines.append("")
    for item in _as_list(report.get("recovery_recommendations")):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Final Report")
    lines.append("")
    lines.append(str(report.get("final_report", "")))
    lines.append("")

    lines.append("---")
    lines.append(f"Generated at: {datetime.utcnow().isoformat()}Z")

    return "\n".join(lines)



def _get_report_scenario(report: Dict[str, Any]) -> str:
    """
    Infer the scenario/report template name from the final GRA report.
    """
    damage = report.get("damage_assessment", {}) or {}
    machine = report.get("machine_readable_summary", {}) or {}

    scenario = (
        damage.get("analysis_mode")
        or machine.get("analysis_mode")
        or report.get("report_template_selected")
        or "generic_damage_report"
    )

    scenario = str(scenario or "generic_damage_report").strip()

    allowed = {
        "wildfire_building",
        "pre_post_svi",
        "cross_view_svi_rsi",
        "object_detection",
        "generic_damage_report",
    }

    if scenario not in allowed:
        return "generic_damage_report"

    return scenario


def _safe_filename_component(value: Any) -> str:
    """
    Convert a value into a safe filename component.
    """
    text = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _resolve_report_output_dir(
    report: Dict[str, Any],
    output_dir: str,
) -> Path:
    """
    Store reports in scenario-specific folders.

    Default behavior:
      outputs/reports
    becomes:
      outputs/Scenarios/<scenario>/reports
    """
    scenario = _get_report_scenario(report)

    normalized = str(output_dir).replace("\\", "/").rstrip("/")

    if normalized in {"outputs/reports", "outputs/report"}:
        return Path("outputs") / "Scenarios" / scenario / "reports"

    return Path(output_dir)


def save_reasoning_report(
    report: Dict[str, Any],
    output_dir: str = "outputs/reports",
) -> Dict[str, Any]:
    scenario = _get_report_scenario(report)

    report_dir = _resolve_report_output_dir(
        report=report,
        output_dir=output_dir,
    )

    report_dir.mkdir(parents=True, exist_ok=True)

    case_id = str(report.get("case_id", "unknown_case"))

    safe_case_id = _safe_filename_component(case_id)
    safe_scenario = _safe_filename_component(scenario)

    base_name = f"{safe_scenario}__{safe_case_id}_report"

    json_path = report_dir / f"{base_name}.json"
    md_path = report_dir / f"{base_name}.md"
    html_path = report_dir / f"{base_name}.html"

    report["report_scenario"] = scenario
    report["report_json_path"] = str(json_path)
    report["report_markdown_path"] = str(md_path)
    report["report_html_path"] = str(html_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    markdown = report_to_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    html_report = render_reasoning_report_html(report)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    return report





def reason_over_case(
    gpa_report: Union[str, Dict[str, Any], None] = None,
    gira_report: Union[str, Dict[str, Any], None] = None,
    gdaa_report: Union[str, Dict[str, Any], None] = None,
    case_id: Optional[str] = None,
    output_dir: str = "outputs/reports",
    model_name: Optional[str] = None,
    use_llm: bool = True,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Final HERMES reasoning step.

    It synthesizes:
    - GPA output
    - GIRA output
    - GDAA output

    It returns a final report and optionally saves JSON + Markdown files.
    """
    gpa = parse_json_like(gpa_report)
    gira = parse_json_like(gira_report)
    gdaa = parse_json_like(gdaa_report)

    deterministic_report = build_deterministic_report(
        gpa_report=gpa,
        gira_report=gira,
        gdaa_report=gdaa,
        case_id=case_id,
    )

    final_report = deterministic_report
    selected_model = model_name or GEMINI_REASONING_MODEL

    if use_llm:
        try:
            prompt = build_llm_prompt(
                deterministic_report=deterministic_report,
                gpa_report=gpa,
                gira_report=gira,
                gdaa_report=gdaa,
            )
            llm_report = call_gemini_reasoning(
                prompt=prompt,
                model_name=selected_model,
            )
            final_report = merge_llm_with_deterministic(
                llm_report=llm_report,
                deterministic_report=deterministic_report,
                model_name=selected_model,
            )
        except Exception as e:
            final_report = deterministic_report
            final_report["llm_used"] = False
            final_report["reasoning_model"] = selected_model
            final_report["llm_error"] = str(e)

    if save_report:
        final_report = save_reasoning_report(
            report=final_report,
            output_dir=output_dir,
        )

    return final_report