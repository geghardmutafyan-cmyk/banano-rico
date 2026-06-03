"""
Banano Rico — AI Nutrition Pipeline (Streamlit Web App)
Runs the full pipeline in a browser-friendly, multi-step form.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banano Rico 🍌",
    page_icon="🍌",
    layout="centered",
)

# ── Constants ──────────────────────────────────────────────────────────────────
RAG_FOLDER = Path("rag")
REPORTS_FOLDER = Path("nutrition_reports")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}

RISK_GREEN = "GREEN"
RISK_YELLOW = "YELLOW"
RISK_RED = "RED"

STEP_WELCOME = "welcome"
STEP_MEDICAL = "medical"
STEP_BODY = "body"
STEP_ACTIVITY = "activity"
STEP_DIET = "diet"
STEP_GOAL = "goal"
STEP_REPORT = "report"
STEP_STOPPED = "stopped"

# ── OpenAI client ──────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


# ── Agent prompts ──────────────────────────────────────────────────────────────
MONITORING_SYSTEM_PROMPT = """
You are first_layer_monitoring_agent for an AI nutrition system.

HARD BLOCK (return RED, should_stop_pipeline: true):
Eating disorder, type 1 diabetes, uncontrolled type 2 diabetes, insulin-dependent diabetes,
kidney disease, liver failure, heart failure, recent heart attack/stroke, active cancer treatment,
pregnancy, breastfeeding, adolescent weight-loss request, severe underweight, malnutrition,
anaphylaxis, severe allergies, or unsafe goals.

MEDICAL REVIEW REQUIRED (return YELLOW, should_stop_pipeline: true):
Hypothyroidism, hyperthyroidism, PCOS, insulin resistance, IBS, GERD, chronic gastritis,
epilepsy, migraines, autoimmune disease, depression, anxiety, OCD, ADHD,
warfarin, GLP-1, steroids, antidepressants, or diuretics.

SOFT WARNING (return GREEN_WITH_WARNING, should_stop_pipeline: false):
Vegetarian, vegan, intermittent fasting, keto, lactose intolerance, night shift, athletic training.

Return only valid JSON:
{
  "risk_level": "RED | YELLOW | GREEN | GREEN_WITH_WARNING",
  "reason": "short reason",
  "should_stop_pipeline": true/false
}
"""

CALORIE_BURN_SYSTEM_PROMPT = """
You are calorie_counter_agent for an AI nutrition pipeline.
Estimate the user's approximate daily calories burned (TDEE / maintenance calories).
Use: height (cm), weight (kg), sex (M/F), steps_per_day, lifestyle_assessment (0-10),
sport_count_per_week, sport_type, age.
Return only one approximate integer number. No explanation. No JSON.
"""

CALORIE_INTAKE_SYSTEM_PROMPT = """
You are nutrition_counter_agent for an AI nutrition pipeline.
Estimate the user's approximate daily calorie intake from food and drinks.
Use: meals_per_day, snacks_per_day, breakfast_description, lunch_description,
dinner_description, snack_description, drinks_description,
portion_size_assessment (0-10), restaurant_or_fast_food_times_per_week,
sweet_food_times_per_week.
Return only one approximate integer number. No explanation. No JSON.
"""

REPORT_SYSTEM_PROMPT = """
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
- Respect allergies and strongly disliked foods.
- Avoid extreme calorie deficits, crash dieting, detoxes, or unsafe supplement advice.

Output rules:
- Return a complete report in Markdown.
- The report must have exactly these major sections:
  # Nutrition Report for {username}
  ## 1. Current State Analysis
  ## 2. Weekly Nutrition Plan
  ## 3. Daily Proportions
  ## 4. Notes and Safety Considerations
- Weekly Nutrition Plan must cover Monday through Sunday.
- Daily Proportions must include protein, carbs, fats, vegetables/fruit, and hydration.
"""

REVISION_SYSTEM_PROMPT = """
You revise an existing nutrition report based on user feedback.
Preserve: Markdown format, four major sections, medical/safety boundaries,
allergies, disliked foods, and the selected goal.
Return only the revised full report in Markdown.
"""


# ── RAG loader ─────────────────────────────────────────────────────────────────
def load_rag_context() -> str:
    if not RAG_FOLDER.exists():
        return "No RAG files found."
    chunks = []
    for path in sorted(RAG_FOLDER.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1", errors="ignore").strip()
        if content:
            chunks.append(f"\n--- SOURCE: {path.name} ---\n{content}")
    return "\n".join(chunks) if chunks else "No RAG files found."


# ── Agent calls ────────────────────────────────────────────────────────────────
def run_monitoring_agent(user_text: str) -> dict:
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": MONITORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def run_calorie_burn_agent(record: dict) -> int:
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": CALORIE_BURN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(record, ensure_ascii=False)},
        ],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 2000


def run_calorie_intake_agent(record: dict) -> int:
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": CALORIE_INTAKE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(record, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 2000


def run_report_agent(record: dict, goal: str, rag_context: str) -> str:
    client = get_client()
    payload = {"goal": goal, "user_record": record, "rag_context": rag_context}
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.35,
    )
    return response.choices[0].message.content.strip()


def run_revision_agent(current_report: str, change_request: str, record: dict, goal: str, rag_context: str) -> str:
    client = get_client()
    payload = {
        "goal": goal,
        "user_record": record,
        "rag_context": rag_context,
        "current_report": current_report,
        "requested_changes": change_request,
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


# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": STEP_WELCOME,
        "record": {},
        "report": "",
        "rag_context": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── UI helpers ─────────────────────────────────────────────────────────────────
def header():
    st.markdown("# 🍌 Banano Rico")
    st.markdown("*AI-powered personalized nutrition planning*")
    st.divider()


def go_to(step: str):
    st.session_state.step = step


# ── Steps ──────────────────────────────────────────────────────────────────────
def step_welcome():
    st.markdown("## Welcome!")
    st.markdown(
        "Banano Rico collects a short health and diet survey, screens for medical risk, "
        "estimates your calorie balance, and generates a personalized weekly nutrition plan."
    )
    st.info("⚕️ This tool is for informational purposes only and does not replace medical advice.")

    username = st.text_input("Choose a username to get started", max_chars=80)
    if st.button("Continue →", type="primary") and username.strip():
        st.session_state.record = {
            "user_id": str(uuid.uuid4()),
            "username": username.strip(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        go_to(STEP_MEDICAL)
        st.rerun()


def step_medical():
    st.markdown("## Step 1 — Medical Screening")
    st.markdown(
        "Before we create a nutrition plan, we need to check for any medical conditions "
        "that require a doctor's involvement."
    )

    choice = st.radio(
        "Do you have any of the following: eating disorders, diabetes, kidney/liver disease, "
        "heart conditions, active cancer treatment, pregnancy, or other serious health conditions?",
        options=["No — I don't have any of these", "Yes — I have one or more", "My situation is more complex…"],
        index=0,
    )

    extra_text = ""
    if choice == "My situation is more complex…":
        extra_text = st.text_area(
            "Please describe your situation briefly:",
            max_chars=1000,
            placeholder="e.g. I have PCOS and mild hypothyroidism…",
        )

    if st.button("Continue →", type="primary"):
        record = st.session_state.record

        if choice == "No — I don't have any of these":
            record["risk_level"] = RISK_GREEN
            record["agent_reason"] = "User denied having listed medical conditions."
            record["first_layer_monitoring_boolean"] = 0
            go_to(STEP_BODY)

        elif choice == "Yes — I have one or more":
            record["risk_level"] = RISK_RED
            record["agent_reason"] = "User confirmed having one of the listed medical conditions."
            record["first_layer_monitoring_boolean"] = 1
            record["pipeline_status"] = "STOPPED"
            record["stopped_reason"] = "Medical screening returned RED risk."
            go_to(STEP_STOPPED)

        else:
            if not extra_text.strip():
                st.warning("Please describe your situation before continuing.")
                return
            with st.spinner("Analyzing your health context…"):
                try:
                    result = run_monitoring_agent(extra_text.strip())
                    risk = str(result.get("risk_level", "YELLOW")).upper().replace("GREEN_WITH_WARNING", "GREEN")
                    record["risk_level"] = risk
                    record["agent_reason"] = result.get("reason", "")
                    record["first_layer_monitoring_text"] = extra_text.strip()

                    if result.get("should_stop_pipeline", risk != RISK_GREEN):
                        record["pipeline_status"] = "STOPPED"
                        record["stopped_reason"] = f"Medical screening returned {risk} risk."
                        go_to(STEP_STOPPED)
                    else:
                        go_to(STEP_BODY)
                except Exception as e:
                    st.error(f"Screening agent error: {e}")
                    return

        st.session_state.record = record
        st.rerun()


def step_stopped():
    record = st.session_state.record
    risk = record.get("risk_level", "RED")
    reason = record.get("agent_reason", "")

    if risk == RISK_RED:
        st.error("## ⛔ Pipeline Stopped — Medical Review Required")
        st.markdown(
            "Based on your answers, we recommend consulting a **doctor or registered dietitian** "
            "before starting any structured nutrition plan."
        )
    else:
        st.warning("## ⚠️ Pipeline Stopped — Human Review Recommended")
        st.markdown(
            "Your health context suggests that a personalized nutrition plan should be created "
            "with guidance from a **qualified healthcare professional**."
        )

    if reason:
        st.markdown(f"**Reason:** {reason}")

    st.info("You can still use the tool once you have received professional clearance.")

    if st.button("Start over"):
        st.session_state.step = STEP_WELCOME
        st.session_state.record = {}
        st.rerun()


def step_body():
    st.markdown("## Step 2 — Body Data")
    record = st.session_state.record

    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
    with col2:
        age = st.number_input("Age", min_value=16, max_value=100, value=25)
        sex = st.radio("Biological sex", ["M", "F"], horizontal=True)

    allergies = st.text_input("Allergies (or type 'none')", value="none", max_chars=500)

    if st.button("Continue →", type="primary"):
        record.update({
            "height": height,
            "weight": weight,
            "age": age,
            "sex": sex,
            "allergy_list": allergies.strip(),
        })
        st.session_state.record = record
        go_to(STEP_ACTIVITY)
        st.rerun()


def step_activity():
    st.markdown("## Step 3 — Physical Activity")
    record = st.session_state.record

    steps = st.slider("Average steps per day", min_value=0, max_value=30000, value=7000, step=500)
    lifestyle = st.slider(
        "Lifestyle activity level (0 = very sedentary, 10 = very active)",
        min_value=0, max_value=10, value=4
    )
    sport_count = st.slider("Exercise sessions per week", min_value=0, max_value=14, value=2)
    sport_type = st.text_input("Type of exercise (or 'none')", value="none", max_chars=200)

    if st.button("Continue →", type="primary"):
        record.update({
            "steps_per_day": steps,
            "lifestyle_assessment": lifestyle,
            "sport_count_per_week": sport_count,
            "sport_type": sport_type.strip(),
        })
        st.session_state.record = record
        go_to(STEP_DIET)
        st.rerun()


def step_diet():
    st.markdown("## Step 4 — Diet Survey")
    record = st.session_state.record

    col1, col2 = st.columns(2)
    with col1:
        meals_per_day = st.number_input("Meals per day", min_value=1, max_value=10, value=3)
        snacks_per_day = st.number_input("Snacks per day", min_value=0, max_value=10, value=1)
    with col2:
        portion_size = st.slider("Typical portion size (0 = tiny, 10 = very large)", 0, 10, 5)
        restaurant_freq = st.number_input("Restaurant/fast food times per week", min_value=0, max_value=21, value=2)

    sweet_freq = st.number_input("Sweet foods/desserts times per week", min_value=0, max_value=21, value=3)

    st.markdown("**Describe your typical meals:**")
    breakfast = st.text_area("Breakfast (include portions if possible)", max_chars=1000,
                              placeholder="e.g. 2 scrambled eggs, 1 slice toast, coffee with milk")
    lunch = st.text_area("Lunch", max_chars=1000, placeholder="e.g. chicken breast 150g, rice 100g, salad")
    dinner = st.text_area("Dinner", max_chars=1000, placeholder="e.g. pasta 200g with tomato sauce and ground beef")
    snacks = st.text_area("Snacks (or 'none')", max_chars=500, placeholder="e.g. apple, handful of nuts")
    drinks = st.text_area("Drinks per day (include sugary drinks, alcohol, etc.)", max_chars=500,
                           placeholder="e.g. 2L water, 1 coffee, 1 Coke 330ml")

    st.markdown("**Food preferences:**")
    loved = st.text_input("Foods you love", max_chars=500, placeholder="e.g. pasta, grilled chicken, fruit")
    hated = st.text_input("Foods you strongly dislike or avoid", max_chars=500, placeholder="e.g. eggplant, liver")

    if st.button("Continue →", type="primary"):
        if not breakfast.strip() or not lunch.strip() or not dinner.strip():
            st.warning("Please fill in at least breakfast, lunch, and dinner.")
            return
        record.update({
            "meals_per_day": meals_per_day,
            "snacks_per_day": snacks_per_day,
            "portion_size_assessment": portion_size,
            "restaurant_or_fast_food_times_per_week": restaurant_freq,
            "sweet_food_times_per_week": sweet_freq,
            "breakfast_description": breakfast.strip(),
            "lunch_description": lunch.strip(),
            "dinner_description": dinner.strip(),
            "snack_description": snacks.strip() or "none",
            "drinks_description": drinks.strip() or "water",
            "loved_foods": loved.strip(),
            "hated_foods": hated.strip(),
        })
        st.session_state.record = record
        go_to(STEP_GOAL)
        st.rerun()


def step_goal():
    st.markdown("## Step 5 — Your Nutrition Goal")
    record = st.session_state.record

    goal_choice = st.radio(
        "What is your primary goal?",
        options=["Weight loss", "Muscle gain"],
        index=0,
    )

    if st.button("Generate my nutrition plan 🍌", type="primary"):
        goal = "weight_loss" if goal_choice == "Weight loss" else "muscle_gain"
        record["nutrition_goal"] = goal

        with st.spinner("Estimating your calorie burn…"):
            try:
                burned = run_calorie_burn_agent(record)
                record["calories_burnt_estimated"] = burned
            except Exception as e:
                st.error(f"Calorie burn estimation failed: {e}")
                return

        with st.spinner("Estimating your calorie intake…"):
            try:
                consumed = run_calorie_intake_agent(record)
                record["calories_consumed_estimated"] = consumed
            except Exception as e:
                st.error(f"Calorie intake estimation failed: {e}")
                return

        with st.spinner("Loading nutrition knowledge base…"):
            rag_context = load_rag_context()
            st.session_state.rag_context = rag_context

        with st.spinner("Generating your personalized nutrition report… (this may take 20–30 seconds)"):
            try:
                report = run_report_agent(record, goal, rag_context)
                st.session_state.report = report
                record["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                record["pipeline_status"] = "COMPLETED"
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                return

        st.session_state.record = record
        go_to(STEP_REPORT)
        st.rerun()


def step_report():
    record = st.session_state.record
    report = st.session_state.report

    burned = record.get("calories_burnt_estimated", "—")
    consumed = record.get("calories_consumed_estimated", "—")
    goal = record.get("nutrition_goal", "—").replace("_", " ").title()

    col1, col2, col3 = st.columns(3)
    col1.metric("🔥 Calories Burned/Day", f"{burned} kcal")
    col2.metric("🍽️ Calories Consumed/Day", f"{consumed} kcal")
    col3.metric("🎯 Goal", goal)

    st.divider()
    st.markdown(report)
    st.divider()

    st.download_button(
        label="⬇️ Download Report (Markdown)",
        data=report,
        file_name=f"{record.get('username', 'user')}_nutrition_report.md",
        mime="text/markdown",
    )

    st.markdown("### Want any changes?")
    change_request = st.text_area(
        "Describe what you'd like to adjust (or leave blank and click Done):",
        max_chars=2000,
        placeholder="e.g. Add more vegetarian options on weekdays. Remove seafood.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Revise Report ✏️") and change_request.strip():
            with st.spinner("Revising your report…"):
                try:
                    revised = run_revision_agent(
                        current_report=report,
                        change_request=change_request.strip(),
                        record=record,
                        goal=record.get("nutrition_goal", "weight_loss"),
                        rag_context=st.session_state.rag_context,
                    )
                    st.session_state.report = revised
                    st.rerun()
                except Exception as e:
                    st.error(f"Revision failed: {e}")

    with col_b:
        if st.button("✅ Done — Start New Plan"):
            st.session_state.step = STEP_WELCOME
            st.session_state.record = {}
            st.session_state.report = ""
            st.rerun()


# ── Progress bar ───────────────────────────────────────────────────────────────
STEP_ORDER = [STEP_WELCOME, STEP_MEDICAL, STEP_BODY, STEP_ACTIVITY, STEP_DIET, STEP_GOAL, STEP_REPORT]

def show_progress():
    step = st.session_state.get("step", STEP_WELCOME)
    if step in (STEP_STOPPED,):
        return
    idx = STEP_ORDER.index(step) if step in STEP_ORDER else 0
    labels = ["Welcome", "Medical", "Body", "Activity", "Diet", "Goal", "Report"]
    progress = idx / (len(STEP_ORDER) - 1)
    st.progress(progress, text=f"Step {idx + 1} of {len(STEP_ORDER)}: {labels[idx]}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_state()
    header()
    show_progress()

    step = st.session_state.step

    if step == STEP_WELCOME:
        step_welcome()
    elif step == STEP_MEDICAL:
        step_medical()
    elif step == STEP_STOPPED:
        step_stopped()
    elif step == STEP_BODY:
        step_body()
    elif step == STEP_ACTIVITY:
        step_activity()
    elif step == STEP_DIET:
        step_diet()
    elif step == STEP_GOAL:
        step_goal()
    elif step == STEP_REPORT:
        step_report()

    st.divider()
    st.caption("🍌 Banano Rico · AI Nutrition Pipeline · For informational purposes only · Not a substitute for medical advice")


if __name__ == "__main__":
    main()
