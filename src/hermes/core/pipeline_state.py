import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional


DEFAULT_INTERMEDIATE_DIR = "outputs/intermediate"


def safe_case_id(case_id: str) -> str:
    """
    Convert a case id into a safe filename component.
    """
    case_id = str(case_id or "unknown_case")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)


def split_image_paths(image_paths: Any) -> List[str]:
    """
    Accept list, tuple, single string, or semicolon-separated string.
    """
    if image_paths is None:
        return []

    if isinstance(image_paths, (list, tuple)):
        return [str(p).strip() for p in image_paths if str(p).strip()]

    text = str(image_paths).strip()
    if not text:
        return []

    return [p.strip() for p in text.split(";") if p.strip()]


def _path_stem(path_value: str) -> str:
    """
    Robust stem extraction for Windows-style and POSIX-style paths.
    """
    text = str(path_value)

    if "\\" in text:
        return PureWindowsPath(text).stem

    return Path(text).stem


def infer_case_id(
    case_id: Optional[str] = None,
    image_paths: Any = None,
    image_path: Optional[str] = None,
) -> str:
    """
    Infer the HERMES case id from explicit case_id or image path(s).
    """
    if case_id:
        return safe_case_id(case_id)

    paths = split_image_paths(image_paths)

    if not paths and image_path:
        paths = [str(image_path)]

    if not paths:
        return "unknown_case"

    stems = [_path_stem(p) for p in paths]
    return safe_case_id("__".join(stems))


def intermediate_path(
    agent_key: str,
    case_id: str,
    output_dir: str = DEFAULT_INTERMEDIATE_DIR,
) -> Path:
    """
    Build path like:
    outputs/intermediate/<case_id>_<agent_key>.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return Path(output_dir) / f"{safe_case_id(case_id)}_{agent_key}.json"


def save_agent_output(
    agent_key: str,
    report: Dict[str, Any],
    case_id: Optional[str] = None,
    image_paths: Any = None,
    image_path: Optional[str] = None,
    output_dir: str = DEFAULT_INTERMEDIATE_DIR,
) -> Dict[str, Any]:
    """
    Save an intermediate agent/tool output and return the same report enriched
    with intermediate metadata.
    """
    resolved_case_id = infer_case_id(
        case_id=case_id,
        image_paths=image_paths,
        image_path=image_path,
    )

    path = intermediate_path(
        agent_key=agent_key,
        case_id=resolved_case_id,
        output_dir=output_dir,
    )

    if isinstance(report, dict):
        report["_hermes_case_id"] = resolved_case_id
        report["_hermes_agent_key"] = agent_key
        report["_hermes_intermediate_path"] = str(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def load_agent_output(
    agent_key: str,
    case_id: str,
    output_dir: str = DEFAULT_INTERMEDIATE_DIR,
) -> Dict[str, Any]:
    """
    Load one intermediate report.
    Returns {} if the file does not exist or cannot be parsed.
    """
    path = intermediate_path(
        agent_key=agent_key,
        case_id=case_id,
        output_dir=output_dir,
    )

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}
    except Exception:
        return {}


def load_case_outputs(
    case_id: str,
    output_dir: str = DEFAULT_INTERMEDIATE_DIR,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all HERMES intermediate reports for a case.
    """
    return {
        "gpa_report": load_agent_output("gpa", case_id, output_dir),
        "gira_report": load_agent_output("gira", case_id, output_dir),
        "gdaa_report": load_agent_output("gdaa", case_id, output_dir),
    }