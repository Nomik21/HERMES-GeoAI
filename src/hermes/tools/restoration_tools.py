import json
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from hermes.core.geo_image_restoration import (
    restore_single_image,
    restore_single_image_with_planner,
    restore_image_collection,
)

from hermes.core.pipeline_state import infer_case_id, save_agent_output


class GeoImageRestorationInput(BaseModel):
    """Input schema for the GeoImageRestorationTool."""

    image_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a single input image that may need restoration. "
            "Use this for single-image cases."
        ),
    )

    image_paths: Optional[str] = Field(
        default=None,
        description=(
            "Paths to one or more input images. "
            "For multiple images, provide paths separated by semicolon ';' or comma ','. "
            "Use this for paired-image or collection cases."
        ),
    )

    source_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional source type override for a single image: "
            "'RS' for remote sensing or 'SV' for street-view."
        ),
    )

    source_types: Optional[str] = Field(
        default=None,
        description=(
            "Optional source types for multiple images, separated by semicolon ';' or comma ','. "
            "Example: 'SV;RS' or 'RS;RS'."
        ),
    )

    hazard_type: str = Field(
        default="unknown",
        description="Optional hazard type, e.g. hurricane, flood, earthquake, wildfire, or unknown.",
    )

    output_dir: str = Field(
        default="outputs/restored",
        description="Directory where restored images will be saved.",
    )

    q_thresh: float = Field(
        default=0.35,
        description="Quality threshold below which restoration is automatically triggered.",
    )

    min_delta_q: float = Field(
        default=0.01,
        description="Minimum required Q improvement for accepting the restored image.",
    )

    mode: str = Field(
        default="deterministic",
        description="Restoration mode: 'deterministic' or 'planner'.",
    )


class GeoImageRestorationTool(BaseTool):
    name: str = "geo_image_restoration_tool"

    description: str = (
        "Restores degraded disaster imagery using deterministic image-processing tools. "
        "It supports single-image and multi-image/pair inputs. "
        "In deterministic mode, it uses rule-based IQA and deterministic restoration tools. "
        "In planner mode, it uses an LLM planner to recommend deterministic tools, "
        "then applies them and accepts the result only if the quality score improves. "
        "For paired or collection inputs, it processes each image independently and returns final_paths."
    )

    args_schema: Type[BaseModel] = GeoImageRestorationInput

    def _run(
        self,
        image_path: Optional[str] = None,
        image_paths: Optional[str] = None,
        source_type: Optional[str] = None,
        source_types: Optional[str] = None,
        hazard_type: str = "unknown",
        output_dir: str = "outputs/restored",
        q_thresh: float = 0.35,
        min_delta_q: float = 0.01,
        mode: str = "deterministic",
    ) -> str:
        mode = str(mode).strip().lower()

        # Prefer image_paths when provided, because it supports both single and pair cases.
        selected_paths = image_paths if image_paths else image_path

        if not selected_paths:
            raise ValueError("Either image_path or image_paths must be provided.")

        case_id = infer_case_id(
            image_paths=selected_paths,
            image_path=image_path,
        )

        # Multi-image / pair case:
        # If image_paths is provided and contains a separator, or if source_types is provided,
        # use the collection wrapper.
        is_collection = bool(image_paths) and (
            ";" in str(image_paths)
            or "," in str(image_paths)
            or bool(source_types)
        )

        if is_collection:
            result = restore_image_collection(
                image_paths=selected_paths,
                source_types=source_types,
                hazard_type=hazard_type,
                output_dir=output_dir,
                q_thresh=q_thresh,
                min_delta_q=min_delta_q,
                mode=mode,
                save_restored=True,
            )

        else:
            # Single-image case, preserving the old behavior.
            if mode == "planner":
                result = restore_single_image_with_planner(
                    img_path=selected_paths,
                    source_type_override=source_type or source_types,
                    hazard_type=hazard_type,
                    output_dir=output_dir,
                    q_thresh=q_thresh,
                    min_delta_q=min_delta_q,
                    save_restored=True,
                )

            else:
                result = restore_single_image(
                    img_path=selected_paths,
                    source_type_override=source_type or source_types,
                    output_dir=output_dir,
                    q_thresh=q_thresh,
                    min_delta_q=min_delta_q,
                    save_restored=True,
                )

                result["mode"] = "deterministic"

        result = save_agent_output(
            agent_key="gira",
            report=result,
            case_id=case_id,
            image_paths=selected_paths,
            image_path=image_path,
        )

        return json.dumps(result, ensure_ascii=False, indent=2)