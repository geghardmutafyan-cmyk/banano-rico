"""
Nutrition pipeline with risk management.

Main improvements over the original version:
- Loads .env before importing modules that may create OpenAI clients.
- Validates OPENAI_API_KEY before the pipeline starts.
- Uses explicit GREEN / YELLOW / RED risk levels.
- Fails closed: unclear, missing, malformed, or failed medical screening stops the diet-plan pipeline.
- Saves partial records even when the pipeline stops.
- Wraps LLM-agent calls in safe error handling.
- Adds audit fields: pipeline_status, stopped_reason, created_at_utc, updated_at_utc.
- Avoids importlib.reload in production-style code.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv

# Important: load environment variables before importing modules that create OpenAI clients.
load_dotenv()

# Import project agents only after load_dotenv().
from first_layer_monitoring_agent import run_first_layer_monitoring_agent
from calorie_counter_agent import run_calorie_counter_agent
from nutrition_counter_agent import run_nutrition_counter_agent
from nutrition_report_rag_agent import (
    ask_goal,
    revise_nutrition_report,
    run_nutrition_report_rag_agent,
    save_nutrition_report,
)


DATA_FILE = Path("df_user.csv")
RAG_FOLDER = Path("rag")

RISK_GREEN = "GREEN"
RISK_YELLOW = "YELLOW"
RISK_RED = "RED"

PIPELINE_COMPLETED = "COMPLETED"
PIPELINE_STOPPED = "STOPPED"
PIPELINE_FAILED = "FAILED"

STOP_WORDS = {"no", "n", "nothing", "finish", "done"}

COLUMNS = [
    "user_id",
    "username",
    "created_at_utc",
    "updated_at_utc",
    "pipeline_status",
    "stopped_reason",

    "first_layer_monitoring_boolean",
    "first_layer_monitoring_text",
    "risk_level",
    "agent_reason",

    "height",
    "weight",
    "sex",
    "age",
    "allergy_list",

    "steps_per_day",
    "lifestyle_assessment",
    "sport_count_per_week",
    "sport_type",

    "meals_per_day",
    "snacks_per_day",
    "breakfast_description",
    "lunch_description",
    "dinner_description",
    "snack_description",
    "drinks_description",
    "portion_size_assessment",
    "restaurant_or_fast_food_times_per_week",
    "sweet_food_times_per_week",

    "loved_foods",
    "hated_foods",

    "calories_burnt_estimated",
    "calories_consumed_estimated",
    "nutrition_goal",
    "nutrition_report_path",
]


class PipelineRiskError(Exception):
    """Raised when the pipeline must stop for safety/risk reasons."""


class PipelineConfigurationError(Exception):
    """Raised when the local project configuration is incomplete."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_environment() -> None:
    """Validate required environment/configuration before running the pipeline."""
    if not os.getenv("OPENAI_API_KEY"):
        raise PipelineConfigurationError(
            "OPENAI_API_KEY is missing. Put it in your .env file or set it as an environment variable."
        )

    if not RAG_FOLDER.exists():
        print(f"Warning: RAG folder '{RAG_FOLDER}' does not exist. Report generation may fail.")


def load_users_df(data_file: Path = DATA_FILE) -> pd.DataFrame:
    if data_file.exists():
        df = pd.read_csv(data_file)
        for column in COLUMNS:
            if column not in df.columns:
                df[column] = None
        return df[COLUMNS]

    return pd.DataFrame(columns=COLUMNS)


def save_users_df(df: pd.DataFrame, data_file: Path = DATA_FILE) -> None:
    temp_file = data_file.with_suffix(".tmp")
    df.to_csv(temp_file, index=False)
    temp_file.replace(data_file)


def generate_user_id() -> str:
    return str(uuid.uuid4())


def ask_text(prompt: str, *, allow_empty: bool = False, max_length: int = 1000) -> str:
    while True:
        value = input(prompt).strip()

        if not value and not allow_empty:
            print("This field cannot be empty.")
            continue

        if len(value) > max_length:
            print(f"Please keep the answer under {max_length} characters.")
            continue

        return value


def ask_int(prompt: str, min_value: int | None = None, max_value: int | None = None) -> int:
    while True:
        value = input(prompt).strip()

        try:
            number = int(value)
        except ValueError:
            print("Please enter an integer.")
            continue

        if min_value is not None and number < min_value:
            print(f"Value must not be less than {min_value}.")
            continue

        if max_value is not None and number > max_value:
            print(f"Value must not exceed {max_value}.")
            continue

        return number


def ask_sex() -> str:
    while True:
        sex = input("Enter biological sex for calorie estimation only, M/F: ").strip().upper()
        if sex in {"M", "F"}:
            return sex
        print("Please enter only M or F.")


def username_exists(username: str, df_user: pd.DataFrame) -> bool:
    return username.lower() in df_user["username"].astype(str).str.lower().values


def ask_unique_username(df_user: pd.DataFrame) -> str:
    while True:
        username = ask_text("Enter a unique username: ", max_length=80)

        if username_exists(username, df_user):
            print("This username already exists. Please enter another one.")
            continue

        return username


def ask_first_layer_monitoring_input() -> str:
    print("\nMedical risk screening")
    print("1 — yes, I have one of the listed medical conditions")
    print("0 — no, I do not have any listed medical condition")
    print("If unclear, describe the situation in text.")

    return ask_text(
        "Do you have any listed medical condition? 1 — yes, 0 — no, or describe: ",
        max_length=1000,
    )


def create_empty_user_record(user_id: str, username: str, first_layer_answer: str) -> dict[str, Any]:
    now = utc_now_iso()
    user_record = {column: None for column in COLUMNS}
    user_record["user_id"] = user_id
    user_record["username"] = username
    user_record["created_at_utc"] = now
    user_record["updated_at_utc"] = now
    user_record["pipeline_status"] = None
    user_record["stopped_reason"] = None

    if first_layer_answer in {"0", "1"}:
        user_record["first_layer_monitoring_boolean"] = int(first_layer_answer)
    else:
        user_record["first_layer_monitoring_text"] = first_layer_answer

    return user_record


def register_user(df_user: pd.DataFrame) -> dict[str, Any]:
    username = ask_unique_username(df_user)
    first_layer_answer = ask_first_layer_monitoring_input()
    return create_empty_user_record(
        user_id=generate_user_id(),
        username=username,
        first_layer_answer=first_layer_answer,
    )


def normalize_agent_result(agent_result: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the first-layer monitoring agent output."""
    risk_level = str(agent_result.get("risk_level", "")).upper().strip()
    reason = str(agent_result.get("reason", "No reason returned by agent.")).strip()
    should_stop = bool(agent_result.get("should_stop_pipeline", risk_level != RISK_GREEN))

    if risk_level not in {RISK_GREEN, RISK_YELLOW, RISK_RED}:
        return {
            "risk_level": RISK_YELLOW,
            "reason": f"Invalid risk level returned by agent: {risk_level!r}. Failing closed.",
            "should_stop_pipeline": True,
        }

    # Fail closed: only GREEN can continue. YELLOW means uncertainty and requires human review.
    if risk_level != RISK_GREEN:
        should_stop = True

    return {
        "risk_level": risk_level,
        "reason": reason,
        "should_stop_pipeline": should_stop,
    }


def first_layer_decision(user_record: dict[str, Any]) -> dict[str, Any]:
    boolean_answer = user_record.get("first_layer_monitoring_boolean")

    if boolean_answer == 1:
        user_record["risk_level"] = RISK_RED
        user_record["agent_reason"] = "User confirmed having one of the listed medical conditions."
        user_record["pipeline_status"] = PIPELINE_STOPPED
        user_record["stopped_reason"] = "Medical screening returned RED risk."
        print("\nPipeline stopped. The user should consult a doctor or registered dietitian first.")
        return user_record

    if boolean_answer == 0:
        user_record["risk_level"] = RISK_GREEN
        user_record["agent_reason"] = "User denied having listed medical conditions."
        print("\nInitial screening passed. The pipeline may continue.")
        return user_record

    user_text = user_record.get("first_layer_monitoring_text")
    if not user_text:
        user_record["risk_level"] = RISK_YELLOW
        user_record["agent_reason"] = "Medical screening answer was missing or unclear."
        user_record["pipeline_status"] = PIPELINE_STOPPED
        user_record["stopped_reason"] = "Unclear medical screening."
        print("\nPipeline stopped because medical screening was unclear.")
        return user_record

    try:
        raw_result = run_first_layer_monitoring_agent(user_text)
        agent_result = normalize_agent_result(raw_result)
    except Exception as exc:
        user_record["risk_level"] = RISK_YELLOW
        user_record["agent_reason"] = f"First-layer monitoring agent failed safely: {exc}"
        user_record["pipeline_status"] = PIPELINE_STOPPED
        user_record["stopped_reason"] = "Medical screening agent failed."
        print("\nPipeline stopped because the medical screening agent failed.")
        return user_record

    user_record["risk_level"] = agent_result["risk_level"]
    user_record["agent_reason"] = agent_result["reason"]

    if agent_result["should_stop_pipeline"]:
        user_record["pipeline_status"] = PIPELINE_STOPPED
        user_record["stopped_reason"] = f"Medical screening returned {agent_result['risk_level']} risk."
        print("\nPipeline stopped. Human medical review is recommended before creating a diet plan.")
    else:
        print(f"\nInitial screening completed. Risk level: {agent_result['risk_level']}")

    return user_record


def should_continue_pipeline(user_record: dict[str, Any]) -> bool:
    # Risk-management rule: continue only on explicit GREEN.
    return user_record.get("risk_level") == RISK_GREEN


def ask_body_data(user_record: dict[str, Any]) -> dict[str, Any]:
    print("\nEnter basic body data.")
    user_record["height"] = ask_int("Height in cm: ", min_value=50, max_value=250)
    user_record["weight"] = ask_int("Weight in kg: ", min_value=20, max_value=300)
    user_record["sex"] = ask_sex()
    user_record["age"] = ask_int("Age: ", min_value=1, max_value=120)
    user_record["allergy_list"] = ask_text("List allergies in free text. If none, type 'none': ", max_length=1000)
    return user_record


def ask_activity_data(user_record: dict[str, Any]) -> dict[str, Any]:
    print("\nDescribe physical activity.")
    user_record["steps_per_day"] = ask_int("Average number of steps per day: ", min_value=0, max_value=100000)
    user_record["lifestyle_assessment"] = ask_int("Lifestyle activity level from 0 to 10: ", min_value=0, max_value=10)
    user_record["sport_count_per_week"] = ask_int("How many times per week do you exercise: ", min_value=0, max_value=30)
    user_record["sport_type"] = ask_text("Type of sport. If none, type 'none': ", max_length=500)
    return user_record


def ask_nutrition_survey_data(user_record: dict[str, Any]) -> dict[str, Any]:
    print("\nDescribe average daily food and drink intake.")
    user_record["meals_per_day"] = ask_int("How many meals do you eat per day: ", min_value=0, max_value=15)
    user_record["snacks_per_day"] = ask_int("How many snacks do you eat per day: ", min_value=0, max_value=20)
    user_record["breakfast_description"] = ask_text("Typical breakfast, including approximate portions: ", max_length=1500)
    user_record["lunch_description"] = ask_text("Typical lunch, including approximate portions: ", max_length=1500)
    user_record["dinner_description"] = ask_text("Typical dinner, including approximate portions: ", max_length=1500)
    user_record["snack_description"] = ask_text("Typical snacks, including approximate portions. If none, type 'none': ", max_length=1500)
    user_record["drinks_description"] = ask_text("Typical drinks per day, including sugar, milk, juice, soda, etc.: ", max_length=1500)
    user_record["portion_size_assessment"] = ask_int(
        "Usual portion size from 0 to 10, where 0 is very small and 10 is very large: ",
        min_value=0,
        max_value=10,
    )
    user_record["restaurant_or_fast_food_times_per_week"] = ask_int(
        "How many times per week do you eat restaurant or fast food: ",
        min_value=0,
        max_value=50,
    )
    user_record["sweet_food_times_per_week"] = ask_int(
        "How many times per week do you eat sweet foods or desserts: ",
        min_value=0,
        max_value=50,
    )

    print("\nBehavioral food-preference data for future personalization.")
    user_record["loved_foods"] = ask_text("What foods/meals do you love to eat? ", max_length=1000)
    user_record["hated_foods"] = ask_text("What foods/meals do you hate or strongly dislike? ", max_length=1000)
    return user_record


def safe_agent_call(agent_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise PipelineRiskError(f"{agent_name} failed: {exc}") from exc


def calorie_counter_decision(user_record: dict[str, Any]) -> dict[str, Any]:
    calories = safe_agent_call("calorie_counter_agent", run_calorie_counter_agent, user_record)
    user_record["calories_burnt_estimated"] = calories
    print(f"\nEstimated daily calories burned: {calories} kcal")
    return user_record


def nutrition_counter_decision(user_record: dict[str, Any]) -> dict[str, Any]:
    calories = safe_agent_call("nutrition_counter_agent", run_nutrition_counter_agent, user_record)
    user_record["calories_consumed_estimated"] = calories
    print(f"Estimated daily calories consumed: {calories} kcal")
    return user_record


def nutrition_report_rag_decision(user_record: dict[str, Any]) -> dict[str, Any]:
    goal = ask_goal()
    user_record["nutrition_goal"] = goal

    report = safe_agent_call(
        "nutrition_report_rag_agent",
        run_nutrition_report_rag_agent,
        user_record=user_record,
        goal=goal,
        rag_folder=str(RAG_FOLDER),
    )

    output_path = save_nutrition_report(username=user_record["username"], report=report)
    user_record["nutrition_report_path"] = str(output_path)

    print("\n" + report)
    print(f"\nReport saved to: {output_path}")

    while True:
        change_request = ask_text(
            "\nAnything to change or add? Type your request, or type 'no' to finish: ",
            max_length=2000,
        )

        if change_request.lower() in STOP_WORDS:
            break

        report = safe_agent_call(
            "nutrition_report_revision_agent",
            revise_nutrition_report,
            current_report=report,
            user_change_request=change_request,
            user_record=user_record,
            goal=goal,
            rag_folder=str(RAG_FOLDER),
        )

        output_path = save_nutrition_report(username=user_record["username"], report=report)
        user_record["nutrition_report_path"] = str(output_path)

        print("\n" + report)
        print(f"\nUpdated report saved to: {output_path}")

    return user_record


def append_user_record_to_df(user_record: dict[str, Any], df_user: pd.DataFrame) -> pd.DataFrame:
    user_record["updated_at_utc"] = utc_now_iso()
    df_user.loc[len(df_user)] = {column: user_record.get(column) for column in COLUMNS}
    save_users_df(df_user)
    return df_user


def run_pipeline() -> pd.DataFrame:
    validate_environment()
    df_user = load_users_df()
    user_record: dict[str, Any] | None = None

    try:
        user_record = register_user(df_user)
        user_record = first_layer_decision(user_record)

        if not should_continue_pipeline(user_record):
            if not user_record.get("pipeline_status"):
                user_record["pipeline_status"] = PIPELINE_STOPPED
            if not user_record.get("stopped_reason"):
                user_record["stopped_reason"] = "Pipeline stopped by risk-management gate."
            return append_user_record_to_df(user_record, df_user)

        user_record = ask_body_data(user_record)
        user_record = ask_activity_data(user_record)
        user_record = ask_nutrition_survey_data(user_record)
        user_record = calorie_counter_decision(user_record)
        user_record = nutrition_counter_decision(user_record)
        user_record = nutrition_report_rag_decision(user_record)
        user_record["pipeline_status"] = PIPELINE_COMPLETED
        return append_user_record_to_df(user_record, df_user)

    except PipelineRiskError as exc:
        print(f"\nPipeline stopped safely: {exc}")
        if user_record is not None:
            user_record["pipeline_status"] = PIPELINE_STOPPED
            user_record["stopped_reason"] = str(exc)
            return append_user_record_to_df(user_record, df_user)
        raise

    except Exception as exc:
        print(f"\nPipeline failed unexpectedly: {exc}")
        if user_record is not None:
            user_record["pipeline_status"] = PIPELINE_FAILED
            user_record["stopped_reason"] = str(exc)
            return append_user_record_to_df(user_record, df_user)
        raise


if __name__ == "__main__":
    df_user = run_pipeline()
    print("\nCurrent df_user:")
    print(df_user)
