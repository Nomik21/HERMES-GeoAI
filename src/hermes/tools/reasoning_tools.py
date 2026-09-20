import json
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from hermes.core.geo_reasoning import reason_over_case, save_reasoning_report
from hermes.core.pipeline_state import load_case_outputs


def _is_empty_report(value: Any) -> bool:
    """
    Detect empty reports passed by CrewAI, such as {}, None, "", or "{}".
    """
    if value is None:
        return True

    if isinstance(value, dict):
        return len(value) == 0

    if isinstance(value, str):
        text = value.strip()
        return text in {"", "{}", "null", "None", "none"}

    return False


class GeoReasoningInput(BaseModel):
    """
    Input schema for the HERMES GeoReasoningTool.

    The reports may be passed as dictionaries, JSON strings,
    or markdown-fenced JSON strings. If CrewAI passes empty reports,
    the tool can load intermediate GPA/GIRA/GDAA outputs from disk
    using the provided case_id.
    """

    gpa_report: Optional[Any] = Field(
        default=None,
        description="Structured output from the GeoPerceptionAgent.",
    )

    gira_report: Optional[Any] = Field(
        default=None,
        description="Structured output from the GeoImageRestorationAgent.",
    )

    gdaa_report: Optional[Any] = Field(
        default=None,
        description="Structured output from the GeoDamageAnalysisAgent.",
    )

    case_id: Optional[str] = Field(
        default=None,
        description="Optional case identifier for the final HERMES report.",
    )

    output_dir: str = Field(
        default="outputs/reports",
        description="Directory where JSON, Markdown, and HTML reports will be saved.",
    )

    intermediate_dir: str = Field(
        default="outputs/intermediate",
        description="Directory where intermediate GPA, GIRA, and GDAA reports are stored.",
    )

    model_name: Optional[str] = Field(
        default=None,
        description="Optional Gemini model name for LLM-assisted reasoning.",
    )

    use_llm: bool = Field(
        default=True,
        description="Whether to use LLM-assisted reasoning after deterministic synthesis.",
    )

    save_report: bool = Field(
        default=True,
        description="Whether to save the report as JSON, Markdown, and HTML files.",
    )


class GeoReasoningTool(BaseTool):
    name: str = "geo_reasoning_tool"

    description: str = (
        "Final HERMES reasoning and reporting tool. It synthesizes the outputs "
        "of the GeoPerceptionAgent, GeoImageRestorationAgent, and "
        "GeoDamageAnalysisAgent into a final situational awareness report. "
        "If CrewAI passes empty reports, it loads the saved intermediate "
        "GPA/GIRA/GDAA reports from disk using case_id. It returns a structured "
        "JSON report and saves JSON, Markdown, and HTML report files."
    )

    args_schema: Type[BaseModel] = GeoReasoningInput

    def _run(
        self,
        gpa_report: Optional[Any] = None,
        gira_report: Optional[Any] = None,
        gdaa_report: Optional[Any] = None,
        case_id: Optional[str] = None,
        output_dir: str = "outputs/reports",
        intermediate_dir: str = "outputs/intermediate",
        model_name: Optional[str] = None,
        use_llm: bool = True,
        save_report: bool = True,
    ) -> str:
        """
        Run final HERMES reasoning.

        If CrewAI does not pass previous task outputs correctly, this tool falls
        back to deterministic intermediate JSON files:
          outputs/intermediate/<case_id>_gpa.json
          outputs/intermediate/<case_id>_gira.json
          outputs/intermediate/<case_id>_gdaa.json
        """

        loaded_from_intermediate = False

        if case_id and (
            _is_empty_report(gpa_report)
            or _is_empty_report(gira_report)
            or _is_empty_report(gdaa_report)
        ):
            loaded = load_case_outputs(
                case_id=case_id,
                output_dir=intermediate_dir,
            )

            if _is_empty_report(gpa_report):
                gpa_report = loaded.get("gpa_report", {})

            if _is_empty_report(gira_report):
                gira_report = loaded.get("gira_report", {})

            if _is_empty_report(gdaa_report):
                gdaa_report = loaded.get("gdaa_report", {})

            loaded_from_intermediate = True


        report = reason_over_case(
            gpa_report=gpa_report,
            gira_report=gira_report,
            gdaa_report=gdaa_report,
            case_id=case_id,
            output_dir=output_dir,
            model_name=model_name,
            use_llm=use_llm,
            save_report=save_report,
        )

        report["intermediate_state_used"] = loaded_from_intermediate

        # Re-save after adding tool-level metadata, so JSON/Markdown/HTML
        # reflect the final returned report.
        if save_report:
            report = save_reasoning_report(
            report=report,
            output_dir=output_dir,
        )

        return json.dumps(report, ensure_ascii=False, indent=2)


