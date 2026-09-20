#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from pathlib import Path, PureWindowsPath

from hermes.crew import Hermes

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

DEFAULT_IMAGE_PATH = "test_images/case_001/image_01.png"


def _split_paths(value: str):
    """
    Accept:
      image.png
      image1.png;image2.png
      image1.png,image2.png
    """
    if not value:
        return []

    text = str(value).strip()

    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]

    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]

    return [text]


def _infer_source_type_from_path(path: str) -> str:
    """
    Lightweight filename-based source type inference for CLI testing.
    The GPA still performs the real perception step inside the crew.
    """
    name = Path(path).name.lower()

    if any(k in name for k in ["satellite", "rsi", "_rs", "remote", "aerial"]):
        return "RS"

    if any(k in name for k in ["svi", "_sv", "street", "streetview"]):
        return "SV"

    # Common HERMES pre/post SVI examples.
    if "2023" in name or "2024" in name:
        return "SV"

    return "unknown"


def _infer_source_types(image_paths):
    return [_infer_source_type_from_path(p) for p in image_paths]


def _infer_pred_mode(image_paths, source_types):
    """
    Lightweight routing metadata for GDAA.
    This mirrors the modes used by the GeoPerceptionAgent.
    """
    n = len(image_paths)
    src = [str(s).upper() for s in source_types]

    if n == 1:
        if src[0] in ["SV", "SVI"]:
            return "post_sv_single"
        if src[0] in ["RS", "RSI"]:
            return "post_rs_single"
        return "post_sv_single"

    if n == 2:
        if src[0] in ["SV", "SVI"] and src[1] in ["SV", "SVI"]:
            return "bitemporal_sv"

        if src[0] in ["RS", "RSI"] and src[1] in ["RS", "RSI"]:
            return "bitemporal_rs"

        if set(src) & {"SV", "SVI"} and set(src) & {"RS", "RSI"}:
            return "post_paired_sv_rs"

    return "unknown"

def _path_stem(path_value: str) -> str:
    text = str(path_value)

    if "\\" in text:
        return PureWindowsPath(text).stem

    return Path(text).stem

def _split_image_paths(image_paths_value):
    """
    Convert a CLI image_paths argument to a clean list of paths.
    Supports:
      - single string path
      - ';' separated string
      - ',' separated string
      - list/tuple
    """
    if image_paths_value is None:
        return []

    if isinstance(image_paths_value, (list, tuple)):
        return [str(p).strip() for p in image_paths_value if str(p).strip()]

    text = str(image_paths_value).strip()
    if not text:
        return []

    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]

    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]

    return [text]


def _make_pair_id(image_paths_value):
    """
    Build a stable case_id from one or more image paths.

    Examples:
      single:
        test_images\\a.jpg
        -> a

      pair:
        test_images\\107806219091079_2023.png;
        test_images\\3754051524846539_2024.png
        -> 107806219091079_2023__3754051524846539_2024
    """
    paths = _split_image_paths(image_paths_value)

    if not paths:
        return "case_unknown"

    stems = [_path_stem(p) for p in paths if str(p).strip()]

    if not stems:
        return "case_unknown"

    if len(stems) == 1:
        return stems[0]

    return "__".join(stems)


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in ["true", "1", "yes", "y"]


def build_inputs_from_cli():
    """
    CLI usage examples:

    Single image:
      python -m hermes.main test_images\\165718466200703_SVI.png SV hurricane auto

    Two images:
      python -m hermes.main "test_images\\pre.png;test_images\\post.png" "SV;SV" hurricane auto

    Arguments:
      1 image_paths     required-ish, defaults to DEFAULT_IMAGE_PATH
      2 source_types    optional, e.g. SV or RS;SV
      3 hazard_type     optional, default unknown
      4 analysis_mode   optional, default auto
      5 pred_mode       optional, otherwise inferred
      6 visualize       optional, default true
    """
    image_paths_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    image_paths_list = _split_paths(image_paths_arg)

    source_types_arg = sys.argv[2] if len(sys.argv) > 2 else None
    hazard_type = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    analysis_mode = sys.argv[4] if len(sys.argv) > 4 else "auto"
    pred_mode_arg = sys.argv[5] if len(sys.argv) > 5 else None
    visualize = _str_to_bool(sys.argv[6]) if len(sys.argv) > 6 else True

    if source_types_arg:
        source_types_list = _split_paths(source_types_arg)
    else:
        source_types_list = _infer_source_types(image_paths_list)

    if len(source_types_list) == 1 and len(image_paths_list) > 1:
        source_types_list = source_types_list * len(image_paths_list)

    pred_mode = pred_mode_arg or _infer_pred_mode(image_paths_list, source_types_list)

    image_paths_joined = ";".join(image_paths_list)
    source_types_joined = ";".join(source_types_list)

    first_image_path = image_paths_list[0] if image_paths_list else DEFAULT_IMAGE_PATH

    inputs = {
        "image_paths": image_paths_joined,
        "image_path": first_image_path,
        "source_types": source_types_joined,
        "source_type": source_types_list[0] if source_types_list else "unknown",
        "hazard_type": hazard_type,
        "pred_mode": pred_mode,
        "analysis_mode": analysis_mode,
        "pair_id": _make_pair_id(image_paths_list),
        "visualize": visualize,
        "current_year": str(datetime.now().year),
    }

    return inputs


def run():
    """
    Run the HERMES crew.
    """
    inputs = build_inputs_from_cli()

    print("HERMES inputs:")
    for k, v in inputs.items():
        print(f"  {k}: {v}")

    try:
        result = Hermes().crew().kickoff(inputs=inputs)
        print(result)

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = build_inputs_from_cli()

    try:
        Hermes().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs,
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Hermes().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and return the results.
    """
    inputs = build_inputs_from_cli()

    try:
        Hermes().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs,
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument.")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": str(datetime.now().year),
        "image_paths": trigger_payload.get("image_paths", DEFAULT_IMAGE_PATH),
        "image_path": trigger_payload.get("image_path", DEFAULT_IMAGE_PATH),
        "source_types": trigger_payload.get("source_types", "unknown"),
        "source_type": trigger_payload.get("source_type", "unknown"),
        "hazard_type": trigger_payload.get("hazard_type", "unknown"),
        "pred_mode": trigger_payload.get("pred_mode", "unknown"),
        "analysis_mode": trigger_payload.get("analysis_mode", "auto"),
        "pair_id": trigger_payload.get("pair_id", "trigger_case"),
        "visualize": trigger_payload.get("visualize", True),
    }

    try:
        result = Hermes().crew().kickoff(inputs=inputs)
        print(result)

    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")



if __name__ == "__main__":
    run()