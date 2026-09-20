import base64
import html
import mimetypes
from pathlib import Path
from tkinter import font
from turtle import color, left
from typing import Any, Dict, List, Optional

from networkx import display
from numpy import size


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _path_to_data_uri(path_value: Optional[str]) -> Optional[str]:
    """
    Convert a local image path into a base64 data URI.

    This makes the HTML report self-contained, so the image remains visible
    even if the report is moved to another computer.
    """
    if not path_value:
        return None

    try:
        path = Path(str(path_value))

        if not path.exists():
            return None

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/jpeg"

        image_bytes = path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")

        return f"data:{mime_type};base64,{encoded}"

    except Exception:
        return None


def _render_image_block(path_value: Optional[str], caption: str) -> str:
    if not path_value:
        return ""

    uri = _path_to_data_uri(path_value)
    escaped_path = _escape(path_value)
    escaped_caption = _escape(caption)

    if uri:
        return f"""
        <div class="image-card">
            <img src="{uri}" alt="{escaped_caption}">
            <div class="caption">{escaped_caption}</div>
            <div class="path">{escaped_path}</div>
        </div>
        """

    return f"""
    <div class="image-card missing">
        <div class="missing-box">Image not found</div>
        <div class="caption">{escaped_caption}</div>
        <div class="path">{escaped_path}</div>
    </div>
    """

def _collect_report_images(report: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Collect scenario-aware image paths for the HTML report.

    Preferred source:
      report["report_images"]

    Backward-compatible fallback:
      image_restoration / machine_readable_summary paths
    """

    explicit_images = report.get("report_images")

    if isinstance(explicit_images, list) and explicit_images:
        images: List[Dict[str, str]] = []

        for item in explicit_images:
            if not isinstance(item, dict):
                continue

            path = item.get("path")
            if not path:
                continue

            caption = (
                item.get("label")
                or item.get("caption")
                or item.get("role")
                or "Report Image"
            )

            images.append(
                {
                    "path": str(path),
                    "caption": str(caption),
                }
            )

        if images:
            return images

    images: List[Dict[str, str]] = []

    restoration = report.get("image_restoration", {}) or {}
    machine = report.get("machine_readable_summary", {}) or {}

    final_paths = restoration.get("final_paths") or machine.get("final_paths")
    final_path = restoration.get("final_path") or machine.get("final_path")
    annotated_path = machine.get("annotated_image_path")

    if isinstance(final_paths, list):
        for index, path in enumerate(final_paths, start=1):
            if path:
                images.append(
                    {
                        "path": str(path),
                        "caption": f"Analysis Image {index}",
                    }
                )

    elif final_path:
        images.append(
            {
                "path": str(final_path),
                "caption": "Analysis Image",
            }
        )

    if annotated_path:
        images.append(
            {
                "path": str(annotated_path),
                "caption": "Annotated Detection Output",
            }
        )

    return images



def _render_key_value_table(rows: List[tuple]) -> str:
    html_rows = []

    for key, value in rows:
        html_rows.append(
            f"""
            <tr>
                <th>{_escape(key)}</th>
                <td>{_format_cell_value(value)}</td>
            </tr>
            """
        )

    return f"""
    <table class="kv-table">
        <tbody>
            {''.join(html_rows)}
        </tbody>
    </table>
    """


def _render_list(items: Any) -> str:
    values = _as_list(items)

    if not values:
        return "<p class='muted'>No information available.</p>"

    lis = "\n".join(f"<li>{_escape(item)}</li>" for item in values)
    return f"<ul>{lis}</ul>"


def _yes_no(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if value is None:
        return "Unknown"
    return str(value)

def _normalize_bool_for_render(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False

    return None


def _format_metric(value: Any) -> str:
    if value is None:
        return ""

    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _format_cell_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return "<br>".join(_escape(item) for item in value)

    if isinstance(value, dict):
        return _escape(value)

    return _escape(value)

def _active_damage_indicators(objects_detected: Any) -> List[str]:
    """
    Return only positive / active damage indicators from objects_detected.
    Example:
      {"damaged_building": 1, "fallen_tree": 0} -> ["damaged_building"]
    """
    if not isinstance(objects_detected, dict):
        return []

    active = []

    for key, value in objects_detected.items():
        try:
            numeric_value = float(value)
            if numeric_value > 0:
                active.append(str(key))
        except Exception:
            if value:
                active.append(str(key))

    return active


def _compact_list_text(value: Any, fallback: str = "not available") -> str:
    values = _as_list(value)

    cleaned = [
        str(item)
        for item in values
        if item is not None and str(item).strip()
    ]

    if not cleaned:
        return fallback

    return ", ".join(cleaned)

def _dashboard_damage_card(damage: Dict[str, Any]) -> tuple:
    """
    Return scenario-aware dashboard label and value.
    """

    analysis_mode = damage.get("analysis_mode")

    if analysis_mode in {"pre_post_svi", "cross_view_svi_rsi"}:
        indicators = _active_damage_indicators(damage.get("objects_detected"))

        if indicators:
            return (
                "Damage Indicators",
                _compact_list_text(indicators),
            )

        return (
            "Affected Assets",
            _compact_list_text(damage.get("affected_assets")),
        )

    if analysis_mode == "object_detection":
        indicators = _active_damage_indicators(damage.get("objects_detected"))

        if indicators:
            return (
                "Detected Objects",
                _compact_list_text(indicators),
            )

        return (
            "Detected Objects",
            _compact_list_text(damage.get("affected_assets")),
        )

    return (
        "Damage Class",
        damage.get("domain_specific_class") or "not available",
    )

def _severity_css_class(severity: Any) -> str:
    value = str(severity or "").strip().lower()

    if value in {"low", "minor", "mild", "none", "no damage"}:
        return "severity-low"

    if value in {"moderate", "medium"}:
        return "severity-moderate"

    if value in {"high", "severe", "critical", "destroyed"}:
        return "severity-high"

    return ""


def _restoration_status_text(restoration: Dict[str, Any]) -> str:
    needed = restoration.get("needed")
    accepted = restoration.get("accepted")

    needed_values = _as_list(needed)
    accepted_values = _as_list(accepted)

    normalized_needed = [
        _normalize_bool_for_render(item)
        for item in needed_values
        if _normalize_bool_for_render(item) is not None
    ]

    normalized_accepted = [
        _normalize_bool_for_render(item)
        for item in accepted_values
        if _normalize_bool_for_render(item) is not None
    ]

    if normalized_accepted:
        total = len(normalized_accepted)
        accepted_count = sum(1 for item in normalized_accepted if item is True)

        if accepted_count == total:
            return f"Accepted ({accepted_count}/{total})"

        if accepted_count == 0:
            if any(item is True for item in normalized_needed):
                return f"Rejected ({accepted_count}/{total} accepted)"
            return "Not required"

        return f"Partially accepted ({accepted_count}/{total})"

    if normalized_needed:
        if any(item is True for item in normalized_needed):
            return "Required"
        return "Not required"

    return "Unknown"


def _build_executive_summary(report: Dict[str, Any]) -> str:
    damage = report.get("damage_assessment", {}) or {}
    restoration = report.get("image_restoration", {}) or {}

    hazard_type = report.get("type_of_disaster") or "unknown"
    input_mode = report.get("input_mode") or "unknown"
    severity = damage.get("severity") or "unknown"
    analysis_mode = damage.get("analysis_mode") or "unknown"
    confidence = damage.get("confidence")

    restoration_status = _restoration_status_text(restoration)

    mode_description = {
        "post_sv_single": "single post-event street-view imagery",
        "post_rs_single": "single post-event remote-sensing imagery",
        "post_paired_sv_rs": "paired post-event street-view and remote-sensing imagery",
        "bitemporal_sv": "before/after street-view imagery",
        "bitemporal_rs": "before/after remote-sensing imagery",
    }.get(input_mode, input_mode)

    confidence_text = "not reported" if confidence is None else str(confidence)

    if analysis_mode == "pre_post_svi":
        affected_assets = _compact_list_text(
            damage.get("affected_assets"),
            fallback="no structured affected assets reported",
        )

        indicators = _compact_list_text(
            _active_damage_indicators(damage.get("objects_detected")),
            fallback="no structured damage indicators reported",
        )

        return (
            f"The HERMES multi-agent pipeline processed this case as a {hazard_type} "
            f"damage-assessment scenario using {mode_description}. The GeoPerception "
            f"Agent provided the input-mode and hazard-context routing, the "
            f"GeoImageRestoration Agent assessed image quality and determined that "
            f"the restoration status was {restoration_status}, and the "
            f"GeoDamageAnalysis Agent selected the {analysis_mode} workflow. "
            f"The assessment identified affected assets including {affected_assets}, "
            f"with active damage indicators including {indicators}. The final "
            f"normalized HERMES severity level is {severity}, with reported "
            f"confidence {confidence_text}."
        )

    if analysis_mode == "cross_view_svi_rsi":
        affected_assets = _compact_list_text(
            damage.get("affected_assets"),
            fallback="no structured affected assets reported",
        )

        indicators = _compact_list_text(
            _active_damage_indicators(damage.get("objects_detected")),
            fallback="no structured damage indicators reported",
        )

        return (
            f"The HERMES multi-agent pipeline processed this case as a {hazard_type} "
            f"damage-assessment scenario using {mode_description}. The "
            f"GeoDamageAnalysis Agent selected the {analysis_mode} workflow, "
            f"combining cross-view evidence from remote-sensing and street-view "
            f"imagery. The assessment identified affected assets including "
            f"{affected_assets}, with active damage indicators including "
            f"{indicators}. The final normalized HERMES severity level is "
            f"{severity}, with reported confidence {confidence_text}. "
            f"Restoration status was {restoration_status}."
        )

    if analysis_mode == "object_detection":
        indicators = _compact_list_text(
            _active_damage_indicators(damage.get("objects_detected")),
            fallback="no structured detected objects reported",
        )

        return (
            f"The HERMES multi-agent pipeline processed this case as a {hazard_type} "
            f"damage-assessment scenario using {mode_description}. The "
            f"GeoDamageAnalysis Agent selected the {analysis_mode} workflow and "
            f"reported detected damage indicators including {indicators}. "
            f"The final normalized HERMES severity level is {severity}, with "
            f"reported confidence {confidence_text}. Restoration status was "
            f"{restoration_status}."
        )

    domain_class = damage.get("domain_specific_class") or "not available"
    damage_range = damage.get("damage_percent_range") or "not specified"

    return (
        f"The HERMES multi-agent pipeline processed this case as a {hazard_type} "
        f"damage-assessment scenario using {mode_description}. The GeoPerception "
        f"Agent provided the input-mode and hazard-context routing, the "
        f"GeoImageRestoration Agent assessed image quality and determined that "
        f"the restoration status was {restoration_status}, and the "
        f"GeoDamageAnalysis Agent selected the {analysis_mode} workflow. "
        f"The domain-specific damage class is {domain_class}, corresponding to an "
        f"estimated visible damage range of {damage_range}. The final normalized "
        f"HERMES severity level is {severity}, with reported confidence "
        f"{confidence_text}."
    )


def _render_dashboard_summary(report: Dict[str, Any]) -> str:
    damage = report.get("damage_assessment", {}) or {}
    restoration = report.get("image_restoration", {}) or {}

    hazard_type = report.get("type_of_disaster") or "unknown"
    input_mode = report.get("input_mode") or "unknown"
    severity = damage.get("severity") or "unknown"
    confidence = damage.get("confidence")
    analysis_mode = damage.get("analysis_mode") or "unknown"
    damage_card_label, damage_card_value = _dashboard_damage_card(damage)
    restoration_status = _restoration_status_text(restoration)

    confidence_text = "N/A" if confidence is None else str(confidence)
    severity_class = _severity_css_class(severity)

    executive_summary = _build_executive_summary(report)

    return f"""
    <div class="dashboard-strip">
        <div class="dashboard-card">
            <div class="dashboard-label">Disaster</div>
            <div class="dashboard-value">{_escape(hazard_type)}</div>
        </div>

        <div class="dashboard-card">
            <div class="dashboard-label">Severity</div>
            <div class="dashboard-value {severity_class}">{_escape(severity)}</div>
        </div>

        <div class="dashboard-card">
            <div class="dashboard-label">Confidence</div>
            <div class="dashboard-value">{_escape(confidence_text)}</div>
        </div>

        <div class="dashboard-card">
            <div class="dashboard-label">Input Mode</div>
            <div class="dashboard-value">{_escape(input_mode)}</div>
        </div>

        <div class="dashboard-card">
            <div class="dashboard-label">{_escape(damage_card_label)}</div>
            <div class="dashboard-value">{_escape(damage_card_value)}</div>
        </div>

        <div class="dashboard-card">
            <div class="dashboard-label">Restoration</div>
            <div class="dashboard-value">{_escape(restoration_status)}</div>
        </div>
    </div>

    <div class="executive-summary">
        <div class="executive-summary-title">Executive Summary</div>
        {_escape(executive_summary)}
    </div>
    """

def _render_restoration_assessment(restoration: Dict[str, Any]) -> str:
    results = restoration.get("results")

    if not isinstance(results, list):
        results = []

    overview_rows = [
        ("Mode", restoration.get("mode")),
        ("Number of images", restoration.get("num_images")),
        ("Needed", [_yes_no(item) for item in _as_list(restoration.get("needed"))]),
        ("Accepted", [_yes_no(item) for item in _as_list(restoration.get("accepted"))]),
        ("Final paths", restoration.get("final_paths") or restoration.get("final_path")),
        ("Restored candidate paths", restoration.get("restored_paths") or restoration.get("restored_path")),
        ("Q_before", restoration.get("Q_before")),
        ("Q_after", restoration.get("Q_after")),
        ("Delta Q", restoration.get("delta_Q")),
        ("Decision", restoration.get("decision")),
    ]

    overview_html = _render_key_value_table(overview_rows)

    if not results:
        return overview_html

    rows_html = []

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue

        rows_html.append(
            f"""
            <tr>
                <td>{_escape(index)}</td>
                <td>{_escape(item.get("image"))}</td>
                <td>{_escape(item.get("source_type"))}</td>
                <td>{_escape(_yes_no(item.get("needs_restoration")))}</td>
                <td>{_escape(_yes_no(item.get("accepted")))}</td>
                <td>{_escape(_format_metric(item.get("Q_before")))}</td>
                <td>{_escape(_format_metric(item.get("Q_after")))}</td>
                <td>{_escape(_format_metric(item.get("delta_Q")))}</td>
                <td>{_escape(item.get("decision"))}</td>
            </tr>
            """
        )

    per_image_html = f"""
    <h3>Per-image Restoration Results</h3>
    <table class="kv-table">
        <thead>
            <tr>
                <th>Index</th>
                <th>Image</th>
                <th>Source</th>
                <th>Needed</th>
                <th>Accepted</th>
                <th>Q_before</th>
                <th>Q_after</th>
                <th>Delta Q</th>
                <th>Decision</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    """

    return overview_html + per_image_html

def _format_detected_objects(objects_detected: Any) -> str:
    if not isinstance(objects_detected, dict) or not objects_detected:
        return "<p class='muted'>No structured object indicators available.</p>"

    rows = []

    for key, value in objects_detected.items():
        try:
            numeric_value = int(value)
        except Exception:
            numeric_value = value

        rows.append(
            f"""
            <tr>
                <td>{_escape(key)}</td>
                <td>{_escape(numeric_value)}</td>
            </tr>
            """
        )

    return f"""
    <table class="kv-table">
        <thead>
            <tr>
                <th>Indicator</th>
                <th>Detected / Count</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _render_change_evidence(change_evidence: Any) -> str:
    if not isinstance(change_evidence, list) or not change_evidence:
        return "<p class='muted'>No structured change evidence available.</p>"

    rows = []

    for item in change_evidence:
        if not isinstance(item, dict):
            continue

        rows.append(
            f"""
            <tr>
                <td>{_escape(item.get("scene_element"))}</td>
                <td>{_escape(item.get("pre_observation"))}</td>
                <td>{_escape(item.get("post_observation"))}</td>
                <td>{_escape(item.get("damage_interpretation"))}</td>
            </tr>
            """
        )

    return f"""
    <h3>Structured Change Evidence</h3>
    <table class="kv-table">
        <thead>
            <tr>
                <th>Scene element</th>
                <th>Pre-event observation</th>
                <th>Post-event observation</th>
                <th>Damage interpretation</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _render_damage_assessment(
    damage: Dict[str, Any],
    machine: Dict[str, Any],
) -> str:
    analysis_mode = damage.get("analysis_mode")

    common_rows = [
        ("Analysis mode", damage.get("analysis_mode")),
        ("Damage presence", _yes_no(damage.get("damage_presence"))),
        ("Severity", damage.get("severity")),
        ("Overall severity", damage.get("overall_severity")),
        ("Confidence", damage.get("confidence")),
    ]

    if analysis_mode == "pre_post_svi":
        scenario_rows = [
            ("Affected assets", damage.get("affected_assets")),
            ("Pre-image path", damage.get("pre_image_path")),
            ("Post-image path", damage.get("post_image_path")),
            ("Reasoning", damage.get("reasoning")),
            (
                "Confidence calibration notes",
                damage.get("confidence_calibration_notes"),
            ),
        ]

        objects_html = """
        <h3>Detected Damage Indicators</h3>
        """ + _format_detected_objects(damage.get("objects_detected"))

        change_html = _render_change_evidence(damage.get("change_evidence"))

        return (
            _render_key_value_table(common_rows + scenario_rows)
            + objects_html
            + change_html
        )

    if analysis_mode == "cross_view_svi_rsi":
        scenario_rows = [
            ("Affected assets", damage.get("affected_assets")),
            ("Remote-sensing image path", damage.get("rs_image_path")),
            ("Street-view image path", damage.get("sv_image_path")),
            ("Reasoning", damage.get("reasoning")),
            (
                "Confidence calibration notes",
                damage.get("confidence_calibration_notes"),
            ),
        ]

        objects_html = """
        <h3>Detected Damage Indicators</h3>
        """ + _format_detected_objects(damage.get("objects_detected"))

        change_html = _render_change_evidence(damage.get("change_evidence"))

        return (
            _render_key_value_table(common_rows + scenario_rows)
            + objects_html
            + change_html
        )

    if analysis_mode == "wildfire_building":
        scenario_rows = [
            ("Three-level severity", damage.get("three_level_severity")),
            ("Domain-specific class", damage.get("domain_specific_class")),
            ("Damage percent range", damage.get("damage_percent_range")),
            ("Reasoning", damage.get("reasoning")),
        ]

        return _render_key_value_table(common_rows + scenario_rows)

    if analysis_mode == "object_detection":
        scenario_rows = [
            ("Affected assets", damage.get("affected_assets")),
            ("Detected objects", damage.get("objects_detected")),
            ("Annotated image path", machine.get("annotated_image_path")),
            ("Reasoning", damage.get("reasoning")),
        ]

        return _render_key_value_table(common_rows + scenario_rows)

    fallback_rows = [
        ("Three-level severity", damage.get("three_level_severity")),
        ("Domain-specific class", damage.get("domain_specific_class")),
        ("Damage percent range", damage.get("damage_percent_range")),
        ("Annotated image path", machine.get("annotated_image_path")),
    ]

    return _render_key_value_table(common_rows + fallback_rows)

def _render_model_response_summary(report: Dict[str, Any]) -> str:
    damage = report.get("damage_assessment", {}) or {}

    analysis_mode = damage.get("analysis_mode")
    object_summary = report.get("object_recognition_summary")

    common_rows = [
        ("Damage analysis mode", analysis_mode),
        ("Disaster severity classification", damage.get("severity")),
        ("Overall severity", damage.get("overall_severity")),
        ("Three-level severity", damage.get("three_level_severity")),
        ("Confidence", damage.get("confidence")),
    ]

    if analysis_mode == "wildfire_building":
        rows = common_rows + [
            ("Domain-specific class", damage.get("domain_specific_class")),
            ("Damage percent range", damage.get("damage_percent_range")),
            ("Affected assets", damage.get("affected_assets")),
            ("Object recognition summary", object_summary),
        ]
        return _render_key_value_table(rows)

    if analysis_mode in {"pre_post_svi", "cross_view_svi_rsi"}:
        active_indicators = _active_damage_indicators(
            damage.get("objects_detected")
        )

        rows = common_rows + [
            ("Affected assets", damage.get("affected_assets")),
            ("Active damage indicators", active_indicators),
            ("Object recognition summary", object_summary),
        ]
        return _render_key_value_table(rows)

    if analysis_mode == "object_detection":
        active_indicators = _active_damage_indicators(
            damage.get("objects_detected")
        )

        rows = common_rows + [
            ("Detected objects / indicators", active_indicators),
            ("Affected assets", damage.get("affected_assets")),
            ("Object recognition summary", object_summary),
        ]
        return _render_key_value_table(rows)

    rows = common_rows + [
        ("Affected assets", damage.get("affected_assets")),
        ("Object recognition summary", object_summary),
    ]
    return _render_key_value_table(rows)

def render_reasoning_report_html(report: Dict[str, Any]) -> str:
    case_id = report.get("case_id", "unknown_case")
    disaster_type = report.get("type_of_disaster", "unknown")
    input_mode = report.get("input_mode", "unknown")

    restoration = report.get("image_restoration", {}) or {}
    damage = report.get("damage_assessment", {}) or {}
    machine = report.get("machine_readable_summary", {}) or {}

    report_images = _collect_report_images(report)
    image_html = "\n".join(
        _render_image_block(item["path"], item["caption"])
        for item in report_images
    )

    if not image_html:
        image_html = "<p class='muted'>No image paths available in report.</p>"

    summary_rows = [
        ("Type of disaster", disaster_type),
        ("Input mode", input_mode),
        ("Image restoration", _yes_no(restoration.get("accepted"))),
        ("Restoration decision", restoration.get("decision")),
        ("Damage analysis mode", damage.get("analysis_mode")),
        ("Disaster severity classification", damage.get("severity")),
        ("Domain-specific class", damage.get("domain_specific_class")),
        ("Damage percent range", damage.get("damage_percent_range")),
        ("Confidence", damage.get("confidence")),
        ("Object recognition", report.get("object_recognition_summary")),
    ]

    restoration_rows = [
        ("Needed", _yes_no(restoration.get("needed"))),
        ("Accepted", _yes_no(restoration.get("accepted"))),
        ("Final path", restoration.get("final_path")),
        ("Restored path", restoration.get("restored_path")),
        ("Q_before", restoration.get("Q_before")),
        ("Q_after", restoration.get("Q_after")),
        ("Delta Q", restoration.get("delta_Q")),
        ("Decision", restoration.get("decision")),
    ]

    damage_rows = [
        ("Analysis mode", damage.get("analysis_mode")),
        ("Damage presence", _yes_no(damage.get("damage_presence"))),
        ("Severity", damage.get("severity")),
        ("Overall severity", damage.get("overall_severity")),
        ("Three-level severity", damage.get("three_level_severity")),
        ("Domain-specific class", damage.get("domain_specific_class")),
        ("Confidence", damage.get("confidence")),
        ("Annotated image path", machine.get("annotated_image_path")),
    ]

    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HERMES Situational Awareness Report - {_escape(case_id)}</title>
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            margin: 0;
            padding: 0;
            background: #f4f6f8;
            color: #1f2933;
        }}

        .page {{
            max-width: 1200px;
            margin: 28px auto;
            background: #ffffff;
            border: 1px solid #d0d7de;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        }}

        .header {{
            padding: 28px 34px;
            background: #111827;
            color: #ffffff;
        }}

        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            letter-spacing: 0.3px;
        }}

        .header .subtitle {{
            font-size: 15px;
            color: #d1d5db;
        }}

        .content {{
            padding: 28px 34px 36px 34px;
        }}


                .dashboard-strip {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 12px;
            margin-bottom: 26px;
        }}

        .dashboard-card {{
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            padding: 13px 14px;
            border-radius: 10px;
            min-height: 72px;
        }}

        .dashboard-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 7px;
            font-weight: 700;
        }}

        .dashboard-value {{
            font-size: 17px;
            font-weight: 800;
            color: #111827;
            word-break: break-word;
        }}

        .dashboard-value.severity-low {{
            color: #166534;
        }}

        .dashboard-value.severity-moderate {{
            color: #b45309;
        }}

        .dashboard-value.severity-high {{
            color: #991b1b;
        }}

        .executive-summary {{
            border-left: 5px solid #111827;
            background: #f9fafb;
            padding: 18px 20px;
            margin-bottom: 28px;
            line-height: 1.55;
            font-size: 15px;
        }}

        .executive-summary-title {{
            font-weight: 800;
            margin-bottom: 8px;
            color: #111827;
            font-size: 16px;
        }}

        @media (max-width: 1000px) {{
            .dashboard-strip {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}

        @media (max-width: 650px) {{
            .dashboard-strip {{
                grid-template-columns: 1fr;
            }}
        }}

        .grid {{
            display: grid;
            grid-template-columns: 1.05fr 1.25fr;
            gap: 24px;
            align-items: start;
        }}

        .section {{
            margin-bottom: 28px;
        }}

        .section h2 {{
            margin: 0 0 14px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #111827;
            font-size: 20px;
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 18px;
        }}

        .image-card {{
            border: 1px solid #d0d7de;
            background: #ffffff;
            padding: 12px;
            text-align: center;
        }}

        .image-card img {{
            max-width: 100%;
            max-height: 390px;
            object-fit: contain;
            border: 1px solid #e5e7eb;
        }}

        .caption {{
            margin-top: 9px;
            font-weight: 700;
            color: #7f1d1d;
        }}

        .path {{
            margin-top: 5px;
            font-size: 12px;
            color: #6b7280;
            word-break: break-all;
        }}

        .missing-box {{
            height: 160px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f3f4f6;
            color: #6b7280;
            border: 1px dashed #9ca3af;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}

        .kv-table th {{
            width: 34%;
            text-align: left;
            background: #f3f4f6;
            border: 1px solid #cbd5e1;
            padding: 10px 12px;
            vertical-align: top;
            font-weight: 700;
        }}

        .kv-table td {{
            border: 1px solid #cbd5e1;
            padding: 10px 12px;
            vertical-align: top;
            word-break: break-word;
        }}

        .reasoning-box {{
            border: 1px solid #cbd5e1;
            background: #ffffff;
            padding: 16px 18px;
            line-height: 1.55;
            font-size: 15px;
            white-space: pre-line;
        }}

        ul {{
            margin-top: 8px;
            padding-left: 24px;
        }}

        li {{
            margin-bottom: 7px;
            line-height: 1.45;
        }}

        .muted {{
            color: #6b7280;
            font-style: italic;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            background: #e5e7eb;
            border-radius: 999px;
            font-size: 13px;
            margin-right: 6px;
        }}

        .footer {{
            padding: 18px 34px;
            border-top: 1px solid #d0d7de;
            color: #6b7280;
            font-size: 12px;
            background: #f9fafb;
        }}

        @media print {{
            body {{
                background: #ffffff;
            }}
            .page {{
                box-shadow: none;
                margin: 0;
                border: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <div class="header">
            <h1>HERMES Situational Awareness Report</h1>
            <div class="subtitle">
                Case ID: {_escape(case_id)} |
                Disaster: {_escape(disaster_type)} |
                Mode: {_escape(input_mode)}
            </div>
        </div>

        <div class="content">
            {_render_dashboard_summary(report)}

            <div class="grid">
                <div class="section">
                    <h2>Input / Output Images</h2>
                    <div class="image-grid">
                        {image_html}
                    </div>
                </div>

                <div class="section">
                    <h2>Model Response Summary</h2>
                    {_render_model_response_summary(report)}
                </div>
            </div>

            <div class="section">
                <h2>Image Restoration Assessment</h2>
                {_render_restoration_assessment(restoration)}
            </div>

            <div class="section">
                <h2>Damage Assessment</h2>
                 {_render_damage_assessment(damage, machine)}
            </div>

            <div class="section">
                <h2>Evidence Chain</h2>
                {_render_list(report.get("evidence_chain"))}
            </div>

            <div class="section">
                <h2>Severity Rationale</h2>
                <div class="reasoning-box">
                    {_escape(report.get("severity_rationale"))}
                </div>
            </div>

            <div class="section">
                <h2>Causal Interpretation</h2>
                <div class="reasoning-box">
                    {_escape(report.get("causal_interpretation"))}
                </div>
            </div>

            <div class="section">
                <h2>Uncertainties and Limitations</h2>
                {_render_list(report.get("uncertainties"))}
            </div>

            <div class="section">
                <h2>Recovery / Verification Recommendations</h2>
                {_render_list(report.get("recovery_recommendations"))}
            </div>

            <div class="section">
                <h2>Final Report</h2>
                <div class="reasoning-box">
                    {_escape(report.get("final_report"))}
                </div>
            </div>
        </div>

        <div class="footer">
            Generated by HERMES GeoReasoning Agent. Report is derived from structured GPA, GIRA, and GDAA outputs.
        </div>
    </div>
</body>
</html>
"""
    return html_doc