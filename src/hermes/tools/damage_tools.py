import json
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from hermes.core.geo_damage_analysis import analyze_damage
from hermes.core.pipeline_state import infer_case_id, save_agent_output


_UNKNOWN_VALUES = {"", "unknown", "none", "null", "n/a", "na"}


def _is_unknown_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in _UNKNOWN_VALUES

    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(_is_unknown_value(v) for v in value)

    return False


def _split_semicolon(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    return [v.strip() for v in text.split(";") if v.strip()]


def _count_image_paths(image_path: Any = None, image_paths: Any = None) -> int:
    paths = _split_semicolon(image_paths)

    if not paths and image_path:
        paths = [str(image_path)]

    return max(len(paths), 1)


def _normalize_source_types(
    source_type: Any = None,
    source_types: Any = None,
    image_path: Any = None,
    image_paths: Any = None,
) -> List[str]:
    """
    Resolve conflicts between source_type and source_types.

    Example:
      source_type='SV'
      source_types='unknown'

    becomes:
      ['SV']
    """

    n_images = _count_image_paths(image_path=image_path, image_paths=image_paths)

    singular = None
    if not _is_unknown_value(source_type):
        singular = str(source_type).strip().upper()

    plural_values = _split_semicolon(source_types)

    if not plural_values or all(_is_unknown_value(v) for v in plural_values):
        if singular:
            return [singular] * n_images
        return ["unknown"] * n_images

    normalized: List[str] = []

    for value in plural_values:
        if _is_unknown_value(value) and singular:
            normalized.append(singular)
        else:
            normalized.append(str(value).strip().upper())

    if len(normalized) == 1 and n_images > 1:
        normalized = normalized * n_images

    return normalized


class GeoDamageAnalysisInput(BaseModel):
    """
    Input schema for the HERMES GeoDamageAnalysisTool.

    Supports:
      - single image path
      - multiple image paths separated by ;
      - explicit analysis_mode
      - automatic routing from GPA pred_mode/source_types/hazard_type
    """

    image_path: Optional[str] = Field(
        default=None,
        description="Path to a single input image.",
    )

    image_paths: Optional[str] = Field(
        default=None,
        description=(
            "One or more image paths. For multiple images, separate paths with ';'. "
            "Example: 'pre.png;post.png' or 'rsi.png;svi.png'."
        ),
    )

    source_type: Optional[str] = Field(
        default=None,
        description="Source type for a single image, e.g. 'SV', 'RS', 'SVI', 'RSI'.",
    )

    source_types: Optional[str] = Field(
        default=None,
        description=(
            "Source types for multiple images, separated with ';'. "
            "Example: 'SV;SV' or 'RS;SV'."
        ),
    )

    hazard_type: str = Field(
        default="unknown",
        description="Disaster/hazard type, e.g. hurricane, wildfire, flood, earthquake.",
    )

    pred_mode: Optional[str] = Field(
        default=None,
        description=(
            "Prediction mode from the GeoPerceptionAgent. "
            "Examples: bitemporal_sv, post_paired_sv_rs, post_sv_single, post_rs_single."
        ),
    )

    analysis_mode: str = Field(
        default="auto",
        description=(
            "GDAA scenario selection. Use 'auto' for router-based selection, or one of: "
            "pre_post_svi, cross_view_svi_rsi, wildfire_building, object_detection."
        ),
    )

    pair_id: Optional[str] = Field(
        default=None,
        description="Optional case/pair identifier for traceability.",
    )

    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional Gemini model override. If omitted, HERMES uses the default model "
            "configured in the environment."
        ),
    )

    visualize: bool = Field(
        default=True,
        description=(
            "Whether to generate annotated visualization for object detection. "
            "Only affects the object_detection scenario."
        ),
    )

    detection_targets: Optional[str] = Field(
        default=None,
        description=(
            "Optional custom detection targets for object_detection. "
            "If omitted, default disaster-related objects are used."
        ),
    )


class GeoDamageAnalysisTool(BaseTool):
    name: str = "geo_damage_analysis_tool"

    description: str = (
        "Unified HERMES GeoDamageAnalysis tool. It performs disaster damage analysis "
        "by routing the input to one of four scenarios: pre/post SVI comparison, "
        "cross-view RSI+SVI assessment, wildfire building damage classification, "
        "or zero-shot object detection/localization. It can operate in automatic mode "
        "using pred_mode, source_types, hazard_type, and number of images."
    )

    args_schema: Type[BaseModel] = GeoDamageAnalysisInput

    def _run(
        self,
        image_path: Optional[str] = None,
        image_paths: Optional[str] = None,
        source_type: Optional[str] = None,
        source_types: Optional[str] = None,
        hazard_type: str = "unknown",
        pred_mode: Optional[str] = None,
        analysis_mode: str = "auto",
        pair_id: Optional[str] = None,
        model_name: Optional[str] = None,
        visualize: bool = True,
        detection_targets: Optional[str] = None,
    ) -> str:
        """
        Execute the unified GDAA router and return JSON.
        """

        selected_image_paths = image_paths if image_paths else image_path

        if not selected_image_paths:
            raise ValueError(
                "GeoDamageAnalysisTool requires either image_path or image_paths."
            )
        
        resolved_case_id = infer_case_id(
            case_id=pair_id,
            image_paths=selected_image_paths,
            image_path=image_path,
        )

        normalized_source_types = _normalize_source_types(
            source_type=source_type,
            source_types=source_types,
            image_path=image_path,
            image_paths=image_paths,
        )

        result = analyze_damage(
            image_paths=selected_image_paths,
            source_types=normalized_source_types,
            hazard_type=hazard_type,
            pred_mode=pred_mode,
            analysis_mode=analysis_mode,
            pair_id=resolved_case_id,
            model_name=model_name,
            visualize=visualize,
            detection_targets=detection_targets,
        )

        result = save_agent_output(
            agent_key="gdaa",
            report=result,
            case_id=resolved_case_id,
            image_paths=selected_image_paths,
            image_path=image_path,
        )

        return json.dumps(result, ensure_ascii=False, indent=2)