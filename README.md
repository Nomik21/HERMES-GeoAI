# HERMES

**A Modular Multi-Agent GeoAI Framework for Disaster Damage Assessment and Evidence-Grounded Report Generation**

## Overview

HERMES is a sequential multi-agent GeoAI pipeline for analysing disaster imagery and generating structured, evidence-grounded damage-assessment reports. It accepts single post-event imagery, paired street-view and remote-sensing imagery, and bitemporal before/after image pairs. The system was developed as part of an MSc thesis in GeoInformatics.

The project focuses on modularity, traceability and component-level evaluation. Instead of producing only a final report, HERMES persists intermediate JSON artefacts after each agent stage, making the pipeline inspectable and re-runnable.

## System Architecture

HERMES is composed of four single-responsibility agents executed in sequence:

1. **GeoPerception Agent (GPA)**  
   Classifies the input configuration, predicts the hazard type from a closed set, and flags whether image restoration is needed. GPA uses the OpenAI Responses API.

2. **GeoImageRestoration Agent (GIRA)**  
   Performs optional, quality-gated image restoration. It computes no-reference image-quality features, applies deterministic OpenCV/scikit-image restoration operators when needed, and accepts a restored image only when the measured quality score improves sufficiently.

3. **GeoDamageAnalysis Agent (GDAA)**  
   Routes each case to the appropriate damage-analysis scenario and produces severity estimates and/or object-indicator detections. The implemented scenarios are `pre_post_svi`, `cross_view_svi_rsi`, `wildfire_building`, and `object_detection`.

4. **GeoReasoning Agent (GRA)**  
   Synthesises the upstream structured outputs into a final disaster-response report. It constructs a locked evidence set, produces a severity rationale, records uncertainties, and generates recovery or verification recommendations. Reports are saved as JSON, Markdown and self-contained HTML.

Each agent writes a structured JSON artefact:

    outputs/intermediate/<case_id>_gpa.json
    outputs/intermediate/<case_id>_gira.json
    outputs/intermediate/<case_id>_gdaa.json
    outputs/<case_id>.json / .md / .html

A detailed implementation-level architecture diagram is available at:

    figures/paper/hermes_architecture_full.svg
    figures/paper/hermes_architecture_caption.md

## Repository Structure

    hermes/
    ├── src/hermes/              # Core HERMES package
    │   ├── core/                # GPA, GIRA, GDAA, GRA and pipeline state
    │   ├── tools/               # CrewAI tool wrappers
    │   ├── config/              # agents.yaml and tasks.yaml
    │   ├── crew.py              # CrewAI crew assembly
    │   └── main.py              # CLI entry point
    ├── scripts/evaluation/      # Evaluation scripts
    ├── notebooks/               # Evaluation visualisation notebook
    ├── figures/paper/           # Architecture diagram and caption
    ├── test_images/             # Small example images for demonstration runs
    ├── pyproject.toml           # Project and dependency metadata
    ├── uv.lock                  # Dependency lockfile
    ├── .env.example             # Environment variable template
    ├── .gitignore
    ├── LICENSE
    └── README.md

## Installation

HERMES requires Python 3.10 or later. The project uses `uv` for dependency management.

Install `uv`:

    pip install uv

Install the project dependencies:

    uv sync

## Environment Variables

Copy the environment template:

    cp .env.example .env

Then fill in the required API keys in `.env`.

Required:

    OPENAI_API_KEY
    GEMINI_API_KEY

Optional model configuration variables:

    GEMINI_PLANNER_MODEL
    GEMINI_DAMAGE_MODEL
    GEMINI_DETECTION_MODEL
    GEMINI_REASONING_MODEL

Never commit `.env` or real API keys.

## Running the Pipeline

Example run:

    uv run python -m hermes.main test_images/case_001/image_01.png SV hurricane auto

General form:

    uv run python -m hermes.main <image_paths> [source_types] [hazard_type] [analysis_mode] [pred_mode] [visualize]

Results are written under `outputs/`, including intermediate JSON artefacts, restored images, detections, and final reports.

## Running the Evaluation Scripts

Evaluation scripts are located in:

    scripts/evaluation/

Example commands:

    uv run python scripts/evaluation/run_gpa_hazard_smoke_evaluation.py
    uv run python scripts/evaluation/run_gdaa_severity_evaluation.py
    uv run python scripts/evaluation/run_gra_report_rubric_evaluation.py
    uv run python scripts/evaluation/run_hermes_smoke_evaluation.py

Full-scale evaluation requires the private benchmark dataset and generated evaluation outputs, which are not included in this public repository.

## Excluded Data and Outputs

The following are intentionally excluded from the public repository:

    .env
    data/
    outputs/
    evaluation/
    thesis/
    figures/evaluation_visualizations/
    .venv/

These files and folders are excluded because they contain private data, generated outputs, large artefacts, thesis documents, local environments, or regenerable evaluation results.

## Citation

HERMES was developed as part of an MSc thesis in GeoInformatics:

    Nomikos, N. (2026). HERMES: A Modular Multi-Agent GeoAI Framework for Disaster Damage Assessment and Evidence-Grounded Report Generation. MSc Thesis.

## License

This project is released under the MIT License. See `LICENSE` for details.
