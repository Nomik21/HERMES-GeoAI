import re
from typing import List

from crewai.tools import tool

from hermes.core.geo_perception import (
    DisasterPerceptionAgent,
    DisasterReasoner,
    ModePerceiver,
    TaskPlanner,
    ensure_pil_image,
)

from hermes.core.pipeline_state import infer_case_id, save_agent_output


def _split_image_paths(image_paths: str) -> List[str]:
    """
    Split image paths passed either with ',' or ';'.
    """
    if not image_paths:
        return []

    return [
        p.strip()
        for p in re.split(r"[;,]", image_paths)
        if p.strip()
    ]


@tool("GeoPerceptionTool")
def geo_perception_tool(image_paths: str) -> dict:
    """
    Run the GeoPerception Agent backend on one or two image paths.

    Args:
        image_paths: One or more image paths separated by ',' or ';'.

    Returns:
        A dictionary containing mode/hazard prediction, reasoning text,
        restoration information, downstream tool plan, and intermediate state path.
    """
    paths: List[str] = _split_image_paths(image_paths)

    if not paths:
        raise ValueError("No image paths were provided to GeoPerceptionTool.")

    images = [ensure_pil_image(path) for path in paths]

    case_id = infer_case_id(image_paths=paths)

    agent = DisasterPerceptionAgent(
        mode_perceiver=ModePerceiver(),
        reasoner=DisasterReasoner(),
        task_planner=TaskPlanner(),
    )

    result = agent.run(
        images=images,
        image_paths=paths,
        run_name=case_id,
    )

    result = save_agent_output(
        agent_key="gpa",
        report=result,
        case_id=case_id,
        image_paths=paths,
    )

    return result