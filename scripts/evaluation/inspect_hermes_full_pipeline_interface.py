from pathlib import Path
import inspect
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "evaluation" / "run_hermes_smoke_evaluation.py",
    PROJECT_ROOT / "src" / "hermes" / "main.py",
    PROJECT_ROOT / "src" / "hermes" / "crew.py",
    PROJECT_ROOT / "src" / "hermes" / "pipeline.py",
]

print("HERMES full-pipeline interface inspection")
print("=" * 90)

for path in CANDIDATE_SCRIPTS:
    print("\n" + "-" * 90)
    print(f"FILE: {path}")
    print(f"EXISTS: {path.exists()}")

    if not path.exists():
        continue

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    print("\nLikely CLI / main lines:")
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if (
            "argparse" in low
            or "def main" in low
            or "if __name__" in low
            or "parser.add_argument" in low
            or "def run" in low
            or "crew" in low
            or "kickoff" in low
            or "report" in low
        ):
            print(f"{i:04d}: {line}")

print("\nDONE")
