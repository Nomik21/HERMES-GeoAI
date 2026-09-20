import json
import os
from typing import Any, Dict, Optional

from PIL import Image


# ============= 0. Gemini Damage Analysis Setup =============

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


if load_dotenv is not None:
    load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

GEMINI_DAMAGE_MODEL = (
    os.getenv("GEMINI_DAMAGE_MODEL")
    or os.getenv("GEMINI_PLANNER_MODEL")
    or "gemini-2.5-flash"
)


def get_gemini_damage_client():
    """
    Create a Gemini client for GDAA.

    The API key must come from .env:
      GEMINI_API_KEY=...
    """
    if genai is None or types is None:
        raise RuntimeError(
            "google-genai is not installed. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install google-genai"
        )

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY was not found. "
            "Please check your .env file."
        )

    return genai.Client(api_key=GEMINI_API_KEY)


# ============= 1. Image Loading =============

def load_pil_image(image_path: str, max_size: int = 1024) -> Image.Image:
    """
    Load an image as RGB PIL image and resize it for efficient API transmission.
    """
    if not image_path:
        raise ValueError("image_path is empty.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img.thumbnail([max_size, max_size])

    return img


# ============= 2. JSON Helpers =============

def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse model JSON robustly, including cases where JSON is wrapped
    inside markdown code fences.
    """
    if raw_text is None:
        raise ValueError("Model response text is empty.")

    text = raw_text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


def _to_binary(value, default: int = 0) -> int:
    """
    Normalize model output to 0 or 1.
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0

    if isinstance(value, str):
        v = value.strip().lower()

        if v in ["1", "true", "yes", "present", "visible"]:
            return 1

        if v in ["0", "false", "no", "absent", "not visible"]:
            return 0

    return default


def _normalize_severity(value: Any) -> str:
    """
    Normalize severity label to HERMES canonical labels.
    """
    if value is None:
        return "Unknown"

    text = str(value).strip().lower()

    mapping = {
        "mild": "Mild",
        "moderate": "Moderate",
        "severe": "Severe",
        "unknown": "Unknown",
        "error": "Unknown",
        "blocked": "Unknown",
    }

    return mapping.get(text, "Unknown")


def _normalize_confidence(value: Any) -> float:
    """
    Normalize confidence to [0.0, 1.0].
    """
    try:
        conf = float(value)
    except Exception:
        conf = 0.0

    return max(0.0, min(1.0, conf))


# ============= 3. Scenario 1 Prompt: SVI Pre/Post Pair =============

def build_pre_post_svi_prompt(hazard_type: str = "unknown") -> str:
    """
    Build prompt for GDAA Scenario 1:
    Street View Image pre/post disaster comparison.
    """
    hazard_type = str(hazard_type or "unknown").lower()

    grading_criteria = """
- Mild: Minimal visible change. Some fallen branches or minor debris may be present,
  but infrastructure, roads, vegetation, façades, and buildings remain largely intact
  compared to the pre-disaster baseline.

- Moderate: Noticeable damage. Visible structural damage to roofs or façades,
  significant debris piles, fallen trees blocking paths that were previously clear,
  localized flooding, road obstruction, or clear degradation compared to the baseline.

- Severe: Drastic transformation. Complete or near-complete destruction of buildings,
  massive flooding, roads becoming fully impassable due to debris or fallen trees,
  or major structural collapse compared to the baseline.
"""

    return f"""
You are the GeoDamageAnalysis Agent (GDAA) of HERMES,
a multi-agent geospatial disaster assessment system.

You are provided with TWO Street View Images of the SAME location.

Input structure:
- Image 1: Pre-disaster baseline image.
- Image 2: Post-disaster image.

Hazard type:
- {hazard_type}

Your tasks:
1. Compare the post-disaster image against the pre-disaster baseline.
2. Perform a dedicated building/structure comparison.
3. Assess the severity of visible disaster-related change.
4. Detect disaster-related objects and indicators in the POST-disaster image only.
5. Identify affected assets.
6. Provide concise evidence of the observed changes.
7. Return only strict JSON.

Critical building comparison rule:
- First identify all buildings or man-made structures visible in Image 1.
- Then check whether each of those structures is still visible and structurally recognizable in Image 2.
- If a building visible in Image 1 is missing, collapsed, substantially altered, or no longer structurally recognizable in Image 2, mark:
  damaged_building = 1
  structural_damage = 1
- Do not ignore a missing or collapsed building simply because the road remains clear.
- If the viewpoint or panorama alignment makes the comparison uncertain, still describe this uncertainty in Change_Evidence and lower the confidence.

Severity criteria:
{grading_criteria}

Objects / indicators to detect in the POST-disaster image:
- debris_pile
- fallen_tree
- flooded_road
- damaged_building
- downed_lines
- road_blockage
- vegetation_damage
- roof_damage
- structural_damage

Severity decision rules:
- If the road is clear and vegetation damage is minor, but a building appears missing, collapsed, or structurally altered, the severity should usually be at least "Moderate".
- Use "Severe" only when there is widespread destruction, complete building collapse with extensive debris, fully impassable roads, massive flooding, or multiple major damage indicators.
- Use "Mild" only when all major structures remain intact and only minor debris or minor vegetation change is visible.

Rules:
- Compare Image 2 against Image 1.
- Use only visible evidence.
- Do not infer hidden or internal damage.
- Do not classify normal viewpoint, lighting, or seasonal differences as disaster damage unless there is visible physical evidence.
- If the apparent change is uncertain, lower the confidence.
- Object detection must refer to the post-disaster image only.
- Pay special attention to buildings, roofs, façades, elevated houses, porches, piers, stairs, and visible structural supports.
- Do not output markdown.
- Do not add any text outside JSON.

Return exactly this JSON schema:
{{
  "Predicted_Severity": "Mild" or "Moderate" or "Severe",
  "Confidence_Score": 0.0,
  "Objects_Detected": {{
    "debris_pile": 0,
    "fallen_tree": 0,
    "flooded_road": 0,
    "damaged_building": 0,
    "downed_lines": 0,
    "road_blockage": 0,
    "vegetation_damage": 0,
    "roof_damage": 0,
    "structural_damage": 0
  }},
  "Affected_Assets": [
    "short asset name"
  ],
  "Change_Evidence": [
    {{
      "scene_element": "road / building / vegetation / utility / other",
      "pre_observation": "what is visible in Image 1",
      "post_observation": "what changed in Image 2",
      "damage_interpretation": "why this indicates damage or no major damage"
    }}
  ],
  "Reasoning": "short explanation of the key changes observed from Image 1 to Image 2",
  "Downstream_Notes": "short notes for the GeoReasoningAgent"
}}
"""


# ============= 4. Scenario 1 Normalization =============

def normalize_pre_post_svi_report(
    data: Dict[str, Any],
    pre_image_path: str,
    post_image_path: str,
    hazard_type: str,
    pair_id: Optional[str] = None,
) -> Dict[str, Any]:
    
    severity = _normalize_severity(
        data.get("Predicted_Severity")
        or data.get("overall_severity")
        or data.get("severity")
    )

    confidence = _normalize_confidence(
        data.get("Confidence_Score")
        or data.get("confidence")
        or data.get("Confidence")
        or 0.0
    )

    objects = (
        data.get("Objects_Detected")
        or data.get("objects_detected")
        or data.get("Objects")
        or {}
    )

    if not isinstance(objects, dict):
        objects = {}

    normalized_objects = {
        "debris_pile": _to_binary(objects.get("debris_pile", 0)),
        "fallen_tree": _to_binary(objects.get("fallen_tree", 0)),
        "flooded_road": _to_binary(objects.get("flooded_road", 0)),
        "damaged_building": _to_binary(objects.get("damaged_building", 0)),
        "downed_lines": _to_binary(objects.get("downed_lines", 0)),
        "road_blockage": _to_binary(objects.get("road_blockage", 0)),
        "vegetation_damage": _to_binary(objects.get("vegetation_damage", 0)),
        "roof_damage": _to_binary(objects.get("roof_damage", 0)),
        "structural_damage": _to_binary(objects.get("structural_damage", 0)),
    }

    affected_assets = (
        data.get("Affected_Assets")
        or data.get("affected_assets")
        or []
    )

    if not isinstance(affected_assets, list):
        affected_assets = [str(affected_assets)]

    change_evidence = (
        data.get("Change_Evidence")
        or data.get("change_evidence")
        or []
    )

    if not isinstance(change_evidence, list):
        change_evidence = []

    reasoning = (
        data.get("Reasoning")
        or data.get("reasoning")
        or ""
    )

    downstream_notes = (
        data.get("Downstream_Notes")
        or data.get("downstream_notes")
        or ""
    )

    damage_presence = (
        severity in ["Mild", "Moderate", "Severe"]
        or any(v == 1 for v in normalized_objects.values())
    )

    # ============= Severity / Confidence Calibration =============
    #
    # Panoramic SVI pre/post comparison can be sensitive to viewpoint,
    # lighting, seasonal differences, and partial occlusion.
    # Therefore, when Severe is mainly inferred from building disappearance
    # but the road remains clear and there are no flood/road-obstruction cues,
    # keep the severity but reduce confidence unless multiple independent
    # severe indicators are present.

    severe_indicators = 0

    if normalized_objects["damaged_building"] == 1:
        severe_indicators += 1

    if normalized_objects["structural_damage"] == 1:
        severe_indicators += 1

    if normalized_objects["flooded_road"] == 1:
        severe_indicators += 1

    if normalized_objects["road_blockage"] == 1:
        severe_indicators += 1

    if normalized_objects["downed_lines"] == 1:
        severe_indicators += 1

    if normalized_objects["debris_pile"] == 1:
        severe_indicators += 1

    confidence_calibration_notes = []

    if severity == "Severe":
        if (
            normalized_objects["damaged_building"] == 1
            and normalized_objects["structural_damage"] == 1
            and normalized_objects["flooded_road"] == 0
            and normalized_objects["road_blockage"] == 0
            and normalized_objects["downed_lines"] == 0
        ):
            if confidence > 0.80:
                confidence = 0.80
                confidence_calibration_notes.append(
                    "Confidence capped at 0.80 because Severe classification is mainly based on building disappearance/structural change in panoramic SVI, while road, flooding, and utility disruption indicators are absent."
                )

        if severe_indicators <= 2 and confidence > 0.75:
            confidence = 0.75
            confidence_calibration_notes.append(
                "Confidence capped at 0.75 because few independent severe damage indicators are visible."
            )

    return {
        "analysis_mode": "pre_post_svi",
        "pair_id": pair_id,
        "source_type": "SV",
        "hazard_type": str(hazard_type or "unknown").lower(),
        "pre_image_path": pre_image_path,
        "post_image_path": post_image_path,
        "damage_presence": bool(damage_presence),
        "overall_severity": severity,
        "confidence": confidence,
        "objects_detected": normalized_objects,
        "affected_assets": affected_assets,
        "change_evidence": change_evidence,
        "reasoning": str(reasoning),
        "confidence_calibration_notes": confidence_calibration_notes,
        "downstream_notes": str(downstream_notes),
    }


# ============= 5. Scenario 1 Core Function =============

def analyze_pre_post_svi_pair(
    pre_image_path: str,
    post_image_path: str,
    hazard_type: str = "unknown",
    pair_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_image_size: int = 1024,
) -> Dict[str, Any]:
    """
    GDAA Scenario 1:
    Analyze a pair of Street View Images from the same location:
      - pre-disaster baseline image
      - post-disaster image

    """
    model = model_name or GEMINI_DAMAGE_MODEL

    img_pre = load_pil_image(pre_image_path, max_size=max_image_size)
    img_post = load_pil_image(post_image_path, max_size=max_image_size)

    prompt = build_pre_post_svi_prompt(hazard_type=hazard_type)

    client = get_gemini_damage_client()

    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, img_pre, img_post],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        data = parse_json_response(response.text)

    except Exception as e:
        data = {
            "Predicted_Severity": "Unknown",
            "Confidence_Score": 0.0,
            "Objects_Detected": {},
            "Affected_Assets": [],
            "Change_Evidence": [],
            "Reasoning": f"GDAA pre/post SVI analysis failed: {e}",
            "Downstream_Notes": (
                "Damage analysis failed; downstream reasoning should treat this "
                "result as low-confidence."
            ),
        }

    return normalize_pre_post_svi_report(
        data=data,
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
        hazard_type=hazard_type,
        pair_id=pair_id,
    )


# ============= 6. Scenario 2 Prompt: Cross-view SVI + RSI Pair =============

def build_cross_view_svi_rsi_prompt(hazard_type: str = "unknown") -> str:
    """
    Build prompt for GDAA Scenario 2:
    Cross-view post-disaster damage assessment using:
      - Remote Sensing Image (RSI)
      - Street View Image (SVI)
    """
    hazard_type = str(hazard_type or "unknown").lower()

    grading_criteria = """
- Mild: Clean or mostly intact scenes. No major property damage, no flooded roads,
  no major debris, and at most minor localized disruption such as small branches,
  light vegetation disturbance, or minor roadside debris.

- Moderate: Noticeable disaster impact. Visible fallen trees, debris piles,
  localized building/roof/façade damage, damaged vegetation, localized water impact,
  or partial road obstruction. Damage is meaningful but not widespread catastrophic destruction.

- Severe: Major destruction. Extensive or widespread tree falls, fully flooded or
  impassable roads, collapsed or severely damaged buildings, large debris fields,
  widespread structural damage, or strong agreement between RSI and SVI that the
  area is heavily impacted.
"""

    return f"""
You are the GeoDamageAnalysis Agent (GDAA) of HERMES,
a multi-agent geospatial disaster assessment system.

You are provided with TWO post-disaster images of the SAME location:

Input structure:
- Image 1: Remote Sensing Image, satellite/aerial view.
- Image 2: Street View Image, ground-level view.

Hazard type:
- {hazard_type}

Your tasks:
1. Analyze the Remote Sensing Image for area-level context:
   roofs, vegetation disturbance, debris fields, flooding, access roads, and spatial extent.
2. Analyze the Street View Image for local visible damage:
   debris, fallen trees, flooded road, damaged building, downed lines, road blockage.
3. Fuse both modalities into one final severity label.
4. Detect damage-related objects and indicators.
5. Provide separate evidence from RSI and SVI.
6. State whether both modalities agree.
7. Return only strict JSON.

Severity criteria:
{grading_criteria}

Objects / indicators to detect:
- debris_pile
- fallen_tree
- flooded_road
- damaged_building
- downed_lines
- road_blockage
- vegetation_damage
- roof_damage
- structural_damage
- flooding

Modality rules:
- Use the RSI mainly for spatial extent, roofs, vegetation patterns, flooding context, and area-level damage.
- Use the SVI mainly for street-level objects and directly visible local damage.
- If SVI clearly shows an object, prioritize SVI for that object.
- If RSI and SVI both support significant damage, confidence may be higher.
- If only one modality supports damage and the other is weak, uncertain, or visually inconsistent, lower confidence.
- Do not classify Severe from one weak cue only.
- Do not infer hidden/internal damage.
- Do not treat different lighting, sky, camera angle, or seasonal appearance as disaster damage unless physical evidence is visible.
- Do not output markdown.
- Do not add text outside JSON.

Return exactly this JSON schema:
{{
  "Predicted_Severity": "Mild" or "Moderate" or "Severe",
  "Confidence_Score": 0.0,
  "Objects": {{
    "debris_pile": 0,
    "fallen_tree": 0,
    "flooded_road": 0,
    "damaged_building": 0,
    "downed_lines": 0,
    "road_blockage": 0,
    "vegetation_damage": 0,
    "roof_damage": 0,
    "structural_damage": 0,
    "flooding": 0
  }},
  "Affected_Assets": [
    "short asset name"
  ],
  "RSI_Evidence": [
    {{
      "region": "image region, e.g. center / upper-left / lower-right",
      "observation": "visible evidence from the remote sensing image",
      "damage_interpretation": "why this supports or does not support damage"
    }}
  ],
  "SVI_Evidence": [
    {{
      "scene_element": "road / building / vegetation / utility / other",
      "observation": "visible evidence from the street-view image",
      "damage_interpretation": "why this supports or does not support damage"
    }}
  ],
  "Modality_Agreement": "strong" or "partial" or "weak" or "conflicting",
  "Reasoning": "short explanation citing evidence from both RSI and SVI",
  "Downstream_Notes": "short notes for the GeoReasoningAgent"
}}
"""


# ============= 7. Scenario 2 Normalization =============

def normalize_cross_view_svi_rsi_report(
    data: Dict[str, Any],
    rsi_image_path: str,
    svi_image_path: str,
    hazard_type: str,
    pair_id: Optional[str] = None,
) -> Dict[str, Any]:
    
    severity = _normalize_severity(
        data.get("Predicted_Severity")
        or data.get("overall_severity")
        or data.get("severity")
    )

    confidence = _normalize_confidence(
        data.get("Confidence_Score")
        or data.get("confidence")
        or data.get("Confidence")
        or 0.0
    )

    objects = (
        data.get("Objects")
        or data.get("objects")
        or data.get("Objects_Detected")
        or data.get("objects_detected")
        or {}
    )

    if not isinstance(objects, dict):
        objects = {}

    normalized_objects = {
        "debris_pile": _to_binary(objects.get("debris_pile", 0)),
        "fallen_tree": _to_binary(objects.get("fallen_tree", 0)),
        "flooded_road": _to_binary(objects.get("flooded_road", 0)),
        "damaged_building": _to_binary(objects.get("damaged_building", 0)),
        "downed_lines": _to_binary(objects.get("downed_lines", 0)),
        "road_blockage": _to_binary(objects.get("road_blockage", 0)),
        "vegetation_damage": _to_binary(objects.get("vegetation_damage", 0)),
        "roof_damage": _to_binary(objects.get("roof_damage", 0)),
        "structural_damage": _to_binary(objects.get("structural_damage", 0)),
        "flooding": _to_binary(objects.get("flooding", 0)),
    }

    affected_assets = (
        data.get("Affected_Assets")
        or data.get("affected_assets")
        or []
    )

    if not isinstance(affected_assets, list):
        affected_assets = [str(affected_assets)]

    rsi_evidence = (
        data.get("RSI_Evidence")
        or data.get("rsi_evidence")
        or []
    )

    if not isinstance(rsi_evidence, list):
        rsi_evidence = []

    svi_evidence = (
        data.get("SVI_Evidence")
        or data.get("svi_evidence")
        or []
    )

    if not isinstance(svi_evidence, list):
        svi_evidence = []

    modality_agreement = (
        data.get("Modality_Agreement")
        or data.get("modality_agreement")
        or "weak"
    )

    modality_agreement = str(modality_agreement).strip().lower()

    if modality_agreement not in ["strong", "partial", "weak", "conflicting"]:
        modality_agreement = "weak"

    reasoning = (
        data.get("Reasoning")
        or data.get("reasoning")
        or ""
    )

    downstream_notes = (
        data.get("Downstream_Notes")
        or data.get("downstream_notes")
        or ""
    )

    damage_presence = (
        severity in ["Mild", "Moderate", "Severe"]
        or any(v == 1 for v in normalized_objects.values())
    )

    # ============= Cross-view Confidence Calibration =============
    #
    # Cross-view fusion is more reliable when RSI and SVI agree.
    # If the model predicts Severe but modality agreement is not strong,
    # we keep the severity but cap confidence.
    #
    # This prevents overconfident severe assessments from one weak modality cue.

    confidence_calibration_notes = []

    severe_indicators = 0

    for key in [
        "debris_pile",
        "fallen_tree",
        "flooded_road",
        "damaged_building",
        "downed_lines",
        "road_blockage",
        "vegetation_damage",
        "roof_damage",
        "structural_damage",
        "flooding",
    ]:
        if normalized_objects[key] == 1:
            severe_indicators += 1

    if severity == "Severe":
        if modality_agreement in ["partial", "weak", "conflicting"] and confidence > 0.80:
            confidence = 0.80
            confidence_calibration_notes.append(
                "Confidence capped at 0.80 because Severe classification does not have strong cross-view agreement."
            )

        if severe_indicators <= 2 and confidence > 0.75:
            confidence = 0.75
            confidence_calibration_notes.append(
                "Confidence capped at 0.75 because few independent severe damage indicators are visible."
            )

    if modality_agreement == "conflicting" and confidence > 0.70:
        confidence = 0.70
        confidence_calibration_notes.append(
            "Confidence capped at 0.70 because RSI and SVI evidence are conflicting."
        )

    return {
        "analysis_mode": "cross_view_svi_rsi",
        "pair_id": pair_id,
        "hazard_type": str(hazard_type or "unknown").lower(),
        "rsi_image_path": rsi_image_path,
        "svi_image_path": svi_image_path,
        "damage_presence": bool(damage_presence),
        "overall_severity": severity,
        "confidence": confidence,
        "objects_detected": normalized_objects,
        "affected_assets": affected_assets,
        "rsi_evidence": rsi_evidence,
        "svi_evidence": svi_evidence,
        "modality_agreement": modality_agreement,
        "reasoning": str(reasoning),
        "confidence_calibration_notes": confidence_calibration_notes,
        "downstream_notes": str(downstream_notes),
    }


# ============= 8. Scenario 2 Core Function =============

def analyze_cross_view_svi_rsi_pair(
    rsi_image_path: str,
    svi_image_path: str,
    hazard_type: str = "unknown",
    pair_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_image_size: int = 1024,
) -> Dict[str, Any]:
    """
    GDAA Scenario 2:
    Analyze a cross-view post-disaster pair from the same location:
      - Remote Sensing Image (satellite/aerial)
      - Street View Image (ground-level)

    """
    model = model_name or GEMINI_DAMAGE_MODEL

    img_rsi = load_pil_image(rsi_image_path, max_size=max_image_size)
    img_svi = load_pil_image(svi_image_path, max_size=max_image_size)

    prompt = build_cross_view_svi_rsi_prompt(hazard_type=hazard_type)

    client = get_gemini_damage_client()

    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, img_rsi, img_svi],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        data = parse_json_response(response.text)

    except Exception as e:
        data = {
            "Predicted_Severity": "Unknown",
            "Confidence_Score": 0.0,
            "Objects": {},
            "Affected_Assets": [],
            "RSI_Evidence": [],
            "SVI_Evidence": [],
            "Modality_Agreement": "weak",
            "Reasoning": f"GDAA cross-view SVI/RSI analysis failed: {e}",
            "Downstream_Notes": (
                "Damage analysis failed; downstream reasoning should treat this "
                "result as low-confidence."
            ),
        }

    return normalize_cross_view_svi_rsi_report(
        data=data,
        rsi_image_path=rsi_image_path,
        svi_image_path=svi_image_path,
        hazard_type=hazard_type,
        pair_id=pair_id,
    )

# ============= 9. Scenario 3 Prompt: SVI Wildfire Building Damage =============

def build_wildfire_building_prompt() -> str:
    """
    Build prompt for GDAA Scenario 3:
    Single-image Street View wildfire building damage classification.
    """
    classification_criteria = """
Classify the damage to the PRIMARY residential structure into exactly one of these 5 categories:

1. 0_No_Damage:
   The primary residential structure is intact. No visible wildfire/fire damage.

2. 1_Affected_1_9:
   Very minor cosmetic damage only, such as soot, smoke staining, blistered paint,
   or very light exterior surface effects. The structure remains stable and usable.

3. 2_Minor_10_25:
   Visible non-structural damage, such as broken windows, melted siding,
   damaged exterior cladding, scorched façade elements, or other repairable damage.
   No major structural failure is visible.

4. 3_Major_26_50:
   Significant structural damage, such as partial roof collapse, major wall damage,
   visible interior burn signs, compromised structural elements, or damage that
   makes the building likely uninhabitable.

5. 4_Destroyed_50plus:
   Total or near-total loss. The structure is collapsed, mostly burned out,
   reduced to foundation/chimney, or more than half of the structure is destroyed.
"""

    return f"""
You are the GeoDamageAnalysis Agent (GDAA) of HERMES,
a multi-agent geospatial disaster assessment system.

You are given ONE post-wildfire Street View Image.

Your task:
Classify the wildfire damage level of the PRIMARY residential structure only.

Classification criteria:
{classification_criteria}

Important rules:
- Focus on the primary residential structure.
- Ignore burnt trees, burned vegetation, ash, and landscape damage if the building itself is intact.
- Do not classify general scene devastation as building damage unless it affects the primary structure.
- Do not count debris as building damage unless it appears to come from the primary structure.
- If multiple buildings are visible, classify the main residential structure most central or visually dominant in the image.
- If no primary residential structure is visible, set primary_structure_visible = false and lower confidence.
- Use only visible evidence.
- Do not infer hidden or internal damage.
- If interior damage is not directly visible, state that interior damage is not assessed.
- Never write that unseen interior damage is likely, probable, or expected.
- Do not output markdown.
- Do not add text outside JSON.

Building indicators to assess:
- soot_or_smoke_staining
- broken_windows
- melted_siding
- exterior_wall_damage
- roof_damage
- partial_collapse
- total_collapse
- foundation_or_chimney_only

Return exactly this JSON schema:
{{
  "Predicted_Class": "0_No_Damage" or "1_Affected_1_9" or "2_Minor_10_25" or "3_Major_26_50" or "4_Destroyed_50plus",
  "Confidence": 0.0,
  "primary_structure_visible": true,
  "Building_Indicators": {{
    "soot_or_smoke_staining": 0,
    "broken_windows": 0,
    "melted_siding": 0,
    "exterior_wall_damage": 0,
    "roof_damage": 0,
    "partial_collapse": 0,
    "total_collapse": 0,
    "foundation_or_chimney_only": 0
  }},
  "Affected_Assets": [
    "short asset name"
  ],
  "Building_Evidence": [
    {{
      "building_part": "roof / façade / windows / walls / foundation / other",
      "observation": "visible evidence",
      "damage_interpretation": "why this supports the predicted class"
    }}
  ],
  "Reasoning": "short explanation based only on visible building damage",
  "Downstream_Notes": "short notes for the GeoReasoningAgent"
}}
"""


# ============= 10. Scenario 3 Normalization =============

def _normalize_wildfire_class(value: Any) -> str:
    """
    Normalize wildfire building damage class to canonical 5-class labels.
    """
    if value is None:
        return "Unknown"

    text = str(value).strip()

    allowed = {
        "0_No_Damage": "0_No_Damage",
        "1_Affected_1_9": "1_Affected_1_9",
        "2_Minor_10_25": "2_Minor_10_25",
        "3_Major_26_50": "3_Major_26_50",
        "4_Destroyed_50plus": "4_Destroyed_50plus",
    }

    if text in allowed:
        return allowed[text]

    lower = text.lower()

    aliases = {
        "no_damage": "0_No_Damage",
        "0": "0_No_Damage",
        "none": "0_No_Damage",
        "intact": "0_No_Damage",

        "affected": "1_Affected_1_9",
        "affected_1_9": "1_Affected_1_9",
        "1": "1_Affected_1_9",

        "minor": "2_Minor_10_25",
        "minor_10_25": "2_Minor_10_25",
        "2": "2_Minor_10_25",

        "major": "3_Major_26_50",
        "major_26_50": "3_Major_26_50",
        "3": "3_Major_26_50",

        "destroyed": "4_Destroyed_50plus",
        "destroyed_50plus": "4_Destroyed_50plus",
        "total_loss": "4_Destroyed_50plus",
        "4": "4_Destroyed_50plus",
    }

    return aliases.get(lower, "Unknown")


def _wildfire_class_metadata(wildfire_class: str) -> Dict[str, Any]:
    """
    Map wildfire 5-class label to HERMES-compatible severity metadata.
    """
    mapping = {
        "0_No_Damage": {
            "overall_severity": "None",
            "three_level_severity": "None",
            "damage_percent_range": "0",
            "damage_presence": False,
        },
        "1_Affected_1_9": {
            "overall_severity": "Mild",
            "three_level_severity": "Mild",
            "damage_percent_range": "1-9",
            "damage_presence": True,
        },
        "2_Minor_10_25": {
            "overall_severity": "Minor",
            "three_level_severity": "Moderate",
            "damage_percent_range": "10-25",
            "damage_presence": True,
        },
        "3_Major_26_50": {
            "overall_severity": "Major",
            "three_level_severity": "Severe",
            "damage_percent_range": "26-50",
            "damage_presence": True,
        },
        "4_Destroyed_50plus": {
            "overall_severity": "Destroyed",
            "three_level_severity": "Severe",
            "damage_percent_range": ">50",
            "damage_presence": True,
        },
        "Unknown": {
            "overall_severity": "Unknown",
            "three_level_severity": "Unknown",
            "damage_percent_range": "Unknown",
            "damage_presence": False,
        },
    }

    return mapping.get(wildfire_class, mapping["Unknown"])


def normalize_wildfire_building_report(
    data: Dict[str, Any],
    image_path: str,
    pair_id: Optional[str] = None,
) -> Dict[str, Any]:
    
    wildfire_class = _normalize_wildfire_class(
        data.get("Predicted_Class")
        or data.get("predicted_class")
        or data.get("wildfire_damage_class")
    )

    class_meta = _wildfire_class_metadata(wildfire_class)

    confidence = _normalize_confidence(
        data.get("Confidence")
        or data.get("confidence")
        or 0.0
    )

    primary_structure_visible = data.get("primary_structure_visible", True)

    if isinstance(primary_structure_visible, bool):
        primary_structure_visible = primary_structure_visible
    else:
        primary_structure_visible = (
            str(primary_structure_visible).strip().lower()
            in ["true", "yes", "1", "visible"]
        )

    indicators = (
        data.get("Building_Indicators")
        or data.get("building_indicators")
        or {}
    )

    if not isinstance(indicators, dict):
        indicators = {}

    normalized_indicators = {
        "soot_or_smoke_staining": _to_binary(indicators.get("soot_or_smoke_staining", 0)),
        "broken_windows": _to_binary(indicators.get("broken_windows", 0)),
        "melted_siding": _to_binary(indicators.get("melted_siding", 0)),
        "exterior_wall_damage": _to_binary(indicators.get("exterior_wall_damage", 0)),
        "roof_damage": _to_binary(indicators.get("roof_damage", 0)),
        "partial_collapse": _to_binary(indicators.get("partial_collapse", 0)),
        "total_collapse": _to_binary(indicators.get("total_collapse", 0)),
        "foundation_or_chimney_only": _to_binary(
            indicators.get("foundation_or_chimney_only", 0)
        ),
    }

    affected_assets = (
        data.get("Affected_Assets")
        or data.get("affected_assets")
        or []
    )

    if not isinstance(affected_assets, list):
        affected_assets = [str(affected_assets)]

    building_evidence = (
        data.get("Building_Evidence")
        or data.get("building_evidence")
        or []
    )

    if not isinstance(building_evidence, list):
        building_evidence = []

    reasoning = (
        data.get("Reasoning")
        or data.get("reasoning")
        or ""
    )

    downstream_notes = (
        data.get("Downstream_Notes")
        or data.get("downstream_notes")
        or ""
    )

    # ============= Wildfire Confidence Calibration =============
    #
    # This task is building-centric. The model should not be overconfident
    # when the primary structure is not visible or when a high-damage class
    # is not supported by structural indicators.

    confidence_calibration_notes = []

    if not primary_structure_visible and confidence > 0.60:
        confidence = 0.60
        confidence_calibration_notes.append(
            "Confidence capped at 0.60 because no primary residential structure is clearly visible."
        )

    if wildfire_class == "4_Destroyed_50plus":
        if (
            normalized_indicators["total_collapse"] == 0
            and normalized_indicators["foundation_or_chimney_only"] == 0
            and normalized_indicators["partial_collapse"] == 0
        ):
            if confidence > 0.75:
                confidence = 0.75
                confidence_calibration_notes.append(
                    "Confidence capped at 0.75 because Destroyed class lacks explicit collapse/foundation-only indicators."
                )

    if wildfire_class == "3_Major_26_50":
        if (
            normalized_indicators["roof_damage"] == 0
            and normalized_indicators["partial_collapse"] == 0
            and normalized_indicators["exterior_wall_damage"] == 0
        ):
            if confidence > 0.75:
                confidence = 0.75
                confidence_calibration_notes.append(
                    "Confidence capped at 0.75 because Major class lacks strong visible roof/wall/collapse indicators."
                )

    if wildfire_class == "0_No_Damage":
        if any(v == 1 for v in normalized_indicators.values()) and confidence > 0.70:
            confidence = 0.70
            confidence_calibration_notes.append(
                "Confidence capped at 0.70 because some building damage indicators were detected despite No_Damage class."
            )

    return {
        "analysis_mode": "wildfire_building",
        "pair_id": pair_id,
        "source_type": "SV",
        "hazard_type": "wildfire",
        "image_path": image_path,
        "primary_structure_visible": bool(primary_structure_visible),
        "damage_presence": bool(class_meta["damage_presence"]),
        "overall_severity": class_meta["overall_severity"],
        "three_level_severity": class_meta["three_level_severity"],
        "wildfire_damage_class": wildfire_class,
        "damage_percent_range": class_meta["damage_percent_range"],
        "confidence": confidence,
        "building_damage_indicators": normalized_indicators,
        "affected_assets": affected_assets,
        "building_evidence": building_evidence,
        "reasoning": str(reasoning),
        "confidence_calibration_notes": confidence_calibration_notes,
        "downstream_notes": str(downstream_notes),
    }


# ============= 11. Scenario 3 Core Function =============

def classify_wildfire_building_damage(
    image_path: str,
    pair_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_image_size: int = 1024,
) -> Dict[str, Any]:
    """
    GDAA Scenario 3:
    Classify wildfire damage to the primary residential structure
    in a single Street View Image.

    
    """
    model = model_name or GEMINI_DAMAGE_MODEL

    img = load_pil_image(image_path, max_size=max_image_size)
    prompt = build_wildfire_building_prompt()

    client = get_gemini_damage_client()

    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        data = parse_json_response(response.text)

    except Exception as e:
        data = {
            "Predicted_Class": "Unknown",
            "Confidence": 0.0,
            "primary_structure_visible": False,
            "Building_Indicators": {},
            "Affected_Assets": [],
            "Building_Evidence": [],
            "Reasoning": f"GDAA wildfire building classification failed: {e}",
            "Downstream_Notes": (
                "Wildfire building damage classification failed; downstream reasoning "
                "should treat this result as low-confidence."
            ),
        }

    return normalize_wildfire_building_report(
        data=data,
        image_path=image_path,
        pair_id=pair_id,
    )



# ============= 12. Scenario 4 Prompt: Zero-shot Damage Object Detection =============

def build_damage_object_detection_prompt(
    source_type: str = "SV",
    hazard_type: str = "unknown",
    detection_targets: Optional[str] = None,
) -> str:
    """
    Build prompt for GDAA Scenario 4:
    Zero-shot damage-related object detection and localization.
    """
    source_type = str(source_type or "SV").upper()
    hazard_type = str(hazard_type or "unknown").lower()

    default_targets = """
Detect and localize disaster-related objects belonging to these categories:

1. Damaged Structures:
   - damaged house
   - damaged building
   - destroyed building
   - collapsed structure
   - roof damage
   - broken windows

2. Road Obstructions / Debris & Vegetation:
   - small piles of debris
   - debris pile
   - fallen tree
   - broken branches
   - blocked road
   - partially blocked road

3. Critical Infrastructure:
   - telephone pole
   - leaning utility pole
   - broken utility pole
   - power lines
   - downed lines

4. Flood / Water Impact:
   - flooded road
   - standing water
   - water over road

5. Navigable Terrain:
   - clear road
   - dirt road
   - cleared path
"""

    targets = detection_targets if detection_targets else default_targets

    return f"""
You are the GeoDamageAnalysis Agent (GDAA) of HERMES,
a multi-agent geospatial disaster assessment system.

You are given ONE post-disaster image.

Image metadata:
- source_type = "{source_type}"
- hazard_type = "{hazard_type}"

Your task:
Detect and localize visible disaster-related objects in the image.

Detection targets:
{targets}

Rules:
- Return bounding boxes only for objects that are visibly present.
- Use the most specific label possible.
- Include the disaster category in the label when possible, e.g. "damaged house (Damaged Structures)".
- Use normalized bounding box coordinates from 0 to 1000.
- The box format must be [ymin, xmin, ymax, xmax].
- Do not invent objects.
- Do not detect hidden/internal damage.
- Do not output markdown.
- Do not add text outside JSON.
- Limit to at most 25 detections.
- If no target objects are visible, return an empty JSON list: [].

Return exactly a JSON array.
Each object must have this schema:
[
  {{
    "box_2d": [0, 0, 100, 100],
    "label": "specific object name (Category)",
    "category": "Damaged Structures / Road Obstructions / Critical Infrastructure / Flood Water Impact / Navigable Terrain / Other",
    "mask_polygon": [],
    "confidence": null
  }}
]
"""


# ============= 13. Scenario 4 Parsing / Normalization =============

def parse_json_array_response(raw_text: str):
    """
    Parse a JSON array robustly, including markdown-fenced output.
    """
    if raw_text is None:
        raise ValueError("Model response text is empty.")

    text = raw_text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError("Object detection response must be a JSON array.")

    return data


def _normalize_box_2d(box):
    """
    Normalize and validate box_2d in [ymin, xmin, ymax, xmax] format,
    with values clamped to [0, 1000].
    """
    if not isinstance(box, list) or len(box) != 4:
        return None

    try:
        y1, x1, y2, x2 = [float(v) for v in box]
    except Exception:
        return None

    y1 = max(0.0, min(1000.0, y1))
    x1 = max(0.0, min(1000.0, x1))
    y2 = max(0.0, min(1000.0, y2))
    x2 = max(0.0, min(1000.0, x2))

    if y2 < y1:
        y1, y2 = y2, y1

    if x2 < x1:
        x1, x2 = x2, x1

    if abs(y2 - y1) < 1 or abs(x2 - x1) < 1:
        return None

    return [int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2))]


def _infer_detection_category(label: str, category: Optional[str] = None) -> str:
    """
    Infer canonical detection category from explicit category or label text.
    """
    if category:
        c = str(category).strip()

        allowed = [
            "Damaged Structures",
            "Road Obstructions",
            "Critical Infrastructure",
            "Flood Water Impact",
            "Navigable Terrain",
            "Other",
        ]

        if c in allowed:
            return c

    label_lower = str(label or "").lower()

    if any(k in label_lower for k in ["damaged house", "damaged building", "destroyed", "collapsed", "roof", "broken window"]):
        return "Damaged Structures"

    if any(k in label_lower for k in ["debris", "fallen tree", "broken branches", "blocked road", "obstruction"]):
        return "Road Obstructions"

    if any(k in label_lower for k in ["telephone pole", "utility pole", "power line", "downed line"]):
        return "Critical Infrastructure"

    if any(k in label_lower for k in ["flood", "standing water", "water over road"]):
        return "Flood Water Impact"

    if any(k in label_lower for k in ["clear road", "dirt road", "cleared path", "navigable"]):
        return "Navigable Terrain"

    return "Other"


def _slug_label(label: str) -> str:
    """
    Convert a detection label to a stable count key.
    """
    text = str(label or "unknown").lower()

    # Remove category suffix if present.
    if "(" in text:
        text = text.split("(")[0].strip()

    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")

    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug.strip("_") or "unknown"


def normalize_object_detection_report(
    data,
    image_path: str,
    source_type: str = "SV",
    hazard_type: str = "unknown",
    pair_id: Optional[str] = None,
    annotated_image_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize raw Gemini detection list into HERMES GDAA schema.
    """
    detections = []

    for item in data:
        if not isinstance(item, dict):
            continue

        box = _normalize_box_2d(item.get("box_2d"))

        if box is None:
            continue

        label = str(item.get("label", "unknown object")).strip()
        category = _infer_detection_category(label, item.get("category"))

        mask_polygon = item.get("mask_polygon", [])
        if not isinstance(mask_polygon, list):
            mask_polygon = []

        confidence = item.get("confidence", None)
        if confidence is not None:
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except Exception:
                confidence = None

        detections.append(
            {
                "label": label,
                "category": category,
                "box_2d": box,
                "box_format": "normalized_ymin_xmin_ymax_xmax_0_1000",
                "mask_polygon": mask_polygon,
                "confidence": confidence,
            }
        )

    object_counts = {}
    category_counts = {}

    for det in detections:
        key = _slug_label(det["label"])
        object_counts[key] = object_counts.get(key, 0) + 1

        cat = det["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    spatial_summary = (
        f"Detected {len(detections)} damage-related or access-related objects."
        if detections
        else "No target damage-related objects were detected."
    )

    return {
        "analysis_mode": "object_detection",
        "pair_id": pair_id,
        "model_name": model_name,
        "source_type": str(source_type or "SV").upper(),
        "hazard_type": str(hazard_type or "unknown").lower(),
        "image_path": image_path,
        "num_detections": len(detections),
        "detections": detections,
        "object_counts": object_counts,
        "category_counts": category_counts,
        "spatial_summary": spatial_summary,
        "annotated_image_path": annotated_image_path,
        "downstream_notes": (
            "Use detections as spatial grounding for damage reasoning. "
            "Bounding boxes are model-generated and should be treated as approximate."
        ),
    }


# ============= 14. Scenario 4 Visualization =============

def draw_damage_detections(
    image_path: str,
    detections,
    output_dir: str = "outputs/detections",
    suffix: str = "_detections",
) -> str:
    """
    Draw detection bounding boxes on the image and save an annotated copy.

    box_2d format:
      [ymin, xmin, ymax, xmax], normalized 0-1000.
    """
    from PIL import ImageDraw, ImageFont

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    colors = [
        "red",
        "lime",
        "blue",
        "yellow",
        "orange",
        "magenta",
        "cyan",
        "white",
    ]

    for idx, det in enumerate(detections):
        box = det.get("box_2d")

        if not box or len(box) != 4:
            continue

        y1, x1, y2, x2 = box

        abs_y1 = int(y1 / 1000 * height)
        abs_x1 = int(x1 / 1000 * width)
        abs_y2 = int(y2 / 1000 * height)
        abs_x2 = int(x2 / 1000 * width)

        if abs_x2 < abs_x1:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y2 < abs_y1:
            abs_y1, abs_y2 = abs_y2, abs_y1

        color = colors[idx % len(colors)]
        label = str(det.get("label", "object"))

        draw.rectangle(
            [abs_x1, abs_y1, abs_x2, abs_y2],
            outline=color,
            width=3,
        )

        label_x = abs_x1
        label_y = max(0, abs_y1 - 14)

        # Simple label background.
        text_width = max(60, len(label) * 7)
        draw.rectangle(
            [label_x, label_y, label_x + text_width, label_y + 14],
            fill=color,
        )
        draw.text(
            [label_x + 2, label_y + 1],
            label,
            fill="black",
            font=font,
        )

    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)
    out_path = os.path.join(output_dir, stem + suffix + ".jpg")

    img.save(out_path, quality=95)

    return out_path


# ============= 15. Scenario 4 Core Function =============

def detect_damage_objects_zero_shot(
    image_path: str,
    source_type: str = "SV",
    hazard_type: str = "unknown",
    detection_targets: Optional[str] = None,
    pair_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_image_size: int = 1024,
    visualize: bool = True,
    output_dir: str = "outputs/detections",
) -> Dict[str, Any]:
    """
    GDAA Scenario 4:
    Zero-shot object detection / localization for disaster-related objects.

    This function is the HERMES-compatible equivalent of the
    zero_shot_object_detection_Agent3 notebook.
    """
    model = model_name or os.getenv("GEMINI_DETECTION_MODEL", GEMINI_DAMAGE_MODEL)

    img = load_pil_image(image_path, max_size=max_image_size)
    prompt = build_damage_object_detection_prompt(
        source_type=source_type,
        hazard_type=hazard_type,
        detection_targets=detection_targets,
    )

    system_instruction = (
        "Return bounding boxes as a JSON array with labels. "
        "Never return masks as images or base64. "
        "Never return markdown or code fencing. "
        "Limit to 25 objects."
    )

    client = get_gemini_damage_client()

    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        raw_detections = parse_json_array_response(response.text)

    except Exception as e:
        raw_detections = []
        failure_report = normalize_object_detection_report(
            data=raw_detections,
            image_path=image_path,
            source_type=source_type,
            hazard_type=hazard_type,
            pair_id=pair_id,
            annotated_image_path=None,
            model_name=model,
        )
        failure_report["error"] = f"GDAA object detection failed: {e}"
        return failure_report

    # First normalize without annotated path.
    report = normalize_object_detection_report(
        data=raw_detections,
        image_path=image_path,
        source_type=source_type,
        hazard_type=hazard_type,
        pair_id=pair_id,
        annotated_image_path=None,
        model_name=model,
    )

    annotated_path = None

    if visualize:
        annotated_path = draw_damage_detections(
            image_path=image_path,
            detections=report["detections"],
            output_dir=output_dir,
        )

    report["annotated_image_path"] = annotated_path

    return report



# ============= 16. Unified GDAA Router =============

def _normalize_image_paths_for_gdaa(image_paths):
    """
    Normalize image_paths input for the GDAA router.

    Accepts:
      - single string path
      - semicolon-separated string
      - comma-separated string
      - list/tuple of paths
    """
    if image_paths is None:
        return []

    if isinstance(image_paths, str):
        text = image_paths.strip()

        if ";" in text:
            return [p.strip() for p in text.split(";") if p.strip()]

        if "," in text:
            return [p.strip() for p in text.split(",") if p.strip()]

        return [text]

    if isinstance(image_paths, (list, tuple)):
        return [str(p).strip() for p in image_paths if str(p).strip()]

    return [str(image_paths).strip()]


def _normalize_source_types_for_gdaa(source_types, n: int):
    """
    Normalize source_types input.

    Accepts:
      - None
      - single string: 'SV'
      - semicolon/comma-separated string: 'RS;SV'
      - list/tuple: ['RS', 'SV']
    """
    if source_types is None:
        return ["unknown"] * n

    if isinstance(source_types, str):
        text = source_types.strip()

        if ";" in text:
            values = [s.strip().upper() for s in text.split(";") if s.strip()]
        elif "," in text:
            values = [s.strip().upper() for s in text.split(",") if s.strip()]
        else:
            values = [text.upper()]
    elif isinstance(source_types, (list, tuple)):
        values = [str(s).strip().upper() for s in source_types if str(s).strip()]
    else:
        values = [str(source_types).strip().upper()]

    if len(values) == 1 and n > 1:
        values = values * n

    if len(values) != n:
        values = (values + ["unknown"] * n)[:n]

    return values


def analyze_damage(
    image_paths,
    source_types=None,
    hazard_type: str = "unknown",
    pred_mode: Optional[str] = None,
    analysis_mode: str = "auto",
    pair_id: Optional[str] = None,
    model_name: Optional[str] = None,
    visualize: bool = True,
    detection_targets: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified GeoDamageAnalysis Agent router.

    It dispatches the input to one of the implemented GDAA scenarios:

    Scenario 1:
      pre_post_svi
      input: pre-disaster SVI + post-disaster SVI

    Scenario 2:
      cross_view_svi_rsi
      input: post-disaster RSI + post-disaster SVI

    Scenario 3:
      wildfire_building
      input: single post-wildfire SVI

    Scenario 4:
      object_detection
      input: single post-disaster image, usually SVI

    Routing can be controlled explicitly with analysis_mode, or inferred from
    pred_mode/source_types/hazard_type.
    """
    paths = _normalize_image_paths_for_gdaa(image_paths)
    n = len(paths)

    if n == 0:
        raise ValueError("analyze_damage requires at least one image path.")

    src_types = _normalize_source_types_for_gdaa(source_types, n)

    hazard = str(hazard_type or "unknown").lower()
    mode = str(analysis_mode or "auto").lower()
    pred = str(pred_mode or "").lower()

    # ---------------- Explicit routing ----------------

    if mode in ["pre_post_svi", "prepost_svi", "bitemporal_svi"]:
        if n < 2:
            raise ValueError("pre_post_svi requires two SVI images: pre and post.")

        return analyze_pre_post_svi_pair(
            pre_image_path=paths[0],
            post_image_path=paths[1],
            hazard_type=hazard,
            pair_id=pair_id,
            model_name=model_name,
        )

    if mode in ["cross_view_svi_rsi", "svi_rsi", "rsi_svi", "post_paired_sv_rs"]:
        if n < 2:
            raise ValueError("cross_view_svi_rsi requires one RSI and one SVI image.")

        # Find RSI and SVI by source_types if available.
        rsi_path = None
        svi_path = None

        for p, s in zip(paths, src_types):
            s_upper = str(s).upper()
            if s_upper in ["RS", "RSI", "SAT", "SATELLITE", "AERIAL"]:
                rsi_path = p
            elif s_upper in ["SV", "SVI", "STREET", "STREET_VIEW"]:
                svi_path = p

        # Fallback: assume paths are already [RSI, SVI].
        if rsi_path is None or svi_path is None:
            rsi_path = paths[0]
            svi_path = paths[1]

        return analyze_cross_view_svi_rsi_pair(
            rsi_image_path=rsi_path,
            svi_image_path=svi_path,
            hazard_type=hazard,
            pair_id=pair_id,
            model_name=model_name,
        )

    if mode in ["wildfire_building", "wildfire_svi_building"]:
        return classify_wildfire_building_damage(
            image_path=paths[0],
            pair_id=pair_id,
            model_name=model_name,
        )

    if mode in ["object_detection", "zero_shot_object_detection", "localization"]:
        return detect_damage_objects_zero_shot(
            image_path=paths[0],
            source_type=src_types[0],
            hazard_type=hazard,
            detection_targets=detection_targets,
            pair_id=pair_id,
            model_name=model_name,
            visualize=visualize,
        )

    # ---------------- Auto routing from GPA pred_mode ----------------

    if pred == "bitemporal_sv":
        if n < 2:
            raise ValueError("pred_mode=bitemporal_sv requires two SVI images.")

        return analyze_pre_post_svi_pair(
            pre_image_path=paths[0],
            post_image_path=paths[1],
            hazard_type=hazard,
            pair_id=pair_id,
            model_name=model_name,
        )

    if pred == "post_paired_sv_rs":
        if n < 2:
            raise ValueError("pred_mode=post_paired_sv_rs requires one RSI and one SVI image.")

        rsi_path = None
        svi_path = None

        for p, s in zip(paths, src_types):
            s_upper = str(s).upper()
            if s_upper in ["RS", "RSI", "SAT", "SATELLITE", "AERIAL"]:
                rsi_path = p
            elif s_upper in ["SV", "SVI", "STREET", "STREET_VIEW"]:
                svi_path = p

        if rsi_path is None or svi_path is None:
            # Fallback convention:
            # if GPA passes post_paired_sv_rs as [SV, RS], reorder here.
            if src_types[0] in ["SV", "SVI"] and src_types[1] in ["RS", "RSI"]:
                svi_path = paths[0]
                rsi_path = paths[1]
            else:
                rsi_path = paths[0]
                svi_path = paths[1]

        return analyze_cross_view_svi_rsi_pair(
            rsi_image_path=rsi_path,
            svi_image_path=svi_path,
            hazard_type=hazard,
            pair_id=pair_id,
            model_name=model_name,
        )

    # ---------------- Auto routing from hazard/source count ----------------

    if n == 1 and hazard == "wildfire" and src_types[0] in ["SV", "SVI", "STREET", "STREET_VIEW"]:
        return classify_wildfire_building_damage(
            image_path=paths[0],
            pair_id=pair_id,
            model_name=model_name,
        )

    if n == 1:
        # Generic single post-disaster image:
        # use object localization as the safest generic GDAA output.
        return detect_damage_objects_zero_shot(
            image_path=paths[0],
            source_type=src_types[0],
            hazard_type=hazard,
            detection_targets=detection_targets,
            pair_id=pair_id,
            model_name=model_name,
            visualize=visualize,
        )

    raise ValueError(
        "Could not infer GDAA scenario. Provide analysis_mode explicitly. "
        "Supported values: pre_post_svi, cross_view_svi_rsi, wildfire_building, object_detection."
    )




