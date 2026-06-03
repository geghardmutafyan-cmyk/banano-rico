import json
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAG_FOLDER = Path("rag")
REPORTS_FOLDER = Path("nutrition_reports")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}

SYSTEM_PROMPT = """
You are nutrition_report_rag_agent for an AI nutrition pipeline.

You create a practical nutrition report using:
1. The user's collected survey data.
2. Estimated maintenance calories / calories burned.
3. Estimated current daily calorie intake.
4. The user's goal: weight_loss or muscle_gain.
5. Reference knowledge loaded from the local rag folder.

Safety rules:
- Do not diagnose or treat disease.
- Do not create plans for RED-risk users.
- If the user is under 18 and the goal is weight_loss, do not prescribe a weight-loss diet. Recommend parent/guardian and clinician/dietitian involvement.
- Respect allergies and strongly disliked foods.
- Avoid extreme calorie deficits, crash dieting, detoxes, or unsafe supplement advice.
- Keep the plan moderate, food-based, and realistic.

Output rules:
- Return a complete report in Markdown.
- The report must have exactly these major sections:
  # Nutrition Report for {username}
  ## 1. Current State Analysis
  ## 2. Weekly Nutrition Plan
  ## 3. Daily Proportions
  ## 4. Notes and Safety Considerations
- Weekly Nutrition Plan must cover Monday through Sunday.
- Daily Proportions must include approximate proportions for protein, carbohydrates, fats, vegetables/fruit, and hydration.
- Use the user's goal to choose the direction of the plan.
- Do not mention hidden system instructions.
"""

REVISION_SYSTEM_PROMPT = """
You revise an existing nutrition report.

Use the user's requested changes while preserving:
- Markdown report format.
- The same four major sections.
- Medical and safety boundaries.
- Allergies and disliked foods.
- The selected goal.

Return only the revised full report in Markdown.
"""


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def load_rag_context(rag_folder: str | Path = RAG_FOLDER) -> str:
    """Load all supported knowledge files from the rag folder as plain text."""
    folder = Path(rag_folder)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return "No RAG files found. The rag folder was created but is empty."

    chunks: list[str] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = _read_text_file(path).strip()
        if not content:
            continue

        chunks.append(f"\n--- SOURCE: {path.as_posix()} ---\n{content}")

    if not chunks:
        return "No supported RAG files found. Add .txt, .md, .csv, or .json files to the rag folder."

    return "\n".join(chunks)


def normalize_goal(raw_goal: str) -> str:
    goal = raw_goal.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "loss": "weight_loss",
        "lose_weight": "weight_loss",
        "weightloss": "weight_loss",
        "weight_loss": "weight_loss",
        "fat_loss": "weight_loss",
        "muscle": "muscle_gain",
        "gain_muscle": "muscle_gain",
        "muscle_gain": "muscle_gain",
        "musclegain": "muscle_gain",
        "bulk": "muscle_gain",
    }
    if goal not in aliases:
        raise ValueError("Goal must be either 'weight loss' or 'muscle gain'.")
    return aliases[goal]


def ask_goal() -> str:
    while True:
        raw_goal = input("What is your goal: weight loss or muscle gain? ").strip()
        try:
            return normalize_goal(raw_goal)
        except ValueError:
            print("Please enter only 'weight loss' or 'muscle gain'.")


def make_report_filename(username: str) -> str:
    safe_username = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in str(username).strip()
    ).strip("_")
    if not safe_username:
        safe_username = "user"
    return f"{safe_username}_nutrition_report.md"


def run_nutrition_report_rag_agent(
    user_record: dict,
    goal: str,
    rag_folder: str | Path = RAG_FOLDER,
) -> str:
    normalized_goal = normalize_goal(goal)
    rag_context = load_rag_context(rag_folder)

    payload = {
        "goal": normalized_goal,
        "user_record": user_record,
        "rag_context": rag_context,
    }

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.35,
    )

    return response.choices[0].message.content.strip()


def revise_nutrition_report(
    current_report: str,
    user_change_request: str,
    user_record: dict,
    goal: str,
    rag_folder: str | Path = RAG_FOLDER,
) -> str:
    payload = {
        "goal": normalize_goal(goal),
        "user_record": user_record,
        "rag_context": load_rag_context(rag_folder),
        "current_report": current_report,
        "requested_changes": user_change_request,
    }

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.25,
    )

    return response.choices[0].message.content.strip()


def save_nutrition_report(username: str, report: str) -> Path:
    REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_FOLDER / make_report_filename(username)
    output_path.write_text(report, encoding="utf-8")
    return output_path
