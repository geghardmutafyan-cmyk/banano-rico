# 🍌 Banano Rico — AI Nutrition Pipeline

> A multi-agent LLM pipeline that screens users for medical risk, estimates their calorie balance, and generates personalized weekly nutrition plans using RAG-enhanced knowledge.

---

## What It Does & Who It's For

**Banano Rico** is an AI-powered nutrition assistant designed for healthy adults who want a personalized, evidence-based nutrition plan without the cost of a dietitian. The system:

1. **Screens for medical risk** — a monitoring agent classifies users as GREEN / YELLOW / RED based on their health context. Only GREEN users continue.
2. **Collects lifestyle data** — body metrics, physical activity, and a full dietary survey.
3. **Estimates calorie balance** — two separate LLM agents estimate calories burned and calories consumed.
4. **Generates a weekly nutrition report** — a RAG agent retrieves nutrition knowledge from local files and writes a personalized Markdown report.
5. **Lets users revise their report** — an interactive revision loop allows natural-language follow-up edits.

---

## Demo & Deployment

| Resource | Link |
|---|---|
| 🚀 Live App (Streamlit Cloud) | [banano-rico.streamlit.app](https://banano-rico.streamlit.app) *(deploy instructions below)* |
| 📁 GitHub Repository | `https://github.com/<your-team>/banano-rico` |

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key (gpt-4.1-mini access required)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-team>/banano-rico.git
cd banano-rico

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your key:
#   OPENAI_API_KEY=sk-...
```

### Run the CLI Pipeline

```bash
python pipeline.py
```

### Run the Web App (Streamlit)

```bash
streamlit run app.py
```

---

## Key Features Walkthrough

### 1. Medical Safety Gate
The pipeline starts with a mandatory medical screening. Users who confirm serious contraindications (eating disorders, diabetes, cancer treatment, pregnancy, etc.) receive a `RED` risk level and are directed to a qualified healthcare provider — the pipeline stops completely. Ambiguous answers are sent to an LLM agent that returns `GREEN`, `YELLOW`, or `RED` with a reason.

### 2. Calorie Estimation Agents
Two independent agents estimate:
- **Calories burned** — using height, weight, sex, steps/day, lifestyle score, and sport frequency.
- **Calories consumed** — using meal descriptions, portion sizes, snack habits, and restaurant frequency.

The gap between these numbers shapes the nutrition plan direction.

### 3. RAG-Enhanced Report Generation
The report agent loads all files from the `rag/` folder (`.txt`, `.md`, `.csv`, `.json`) as context before generating the plan. This allows the team to add or update nutrition knowledge without changing the code.

### 4. Interactive Revision Loop
After receiving their report, users can type natural-language change requests (e.g., "add more vegetarian options on weekdays") and receive a revised report. This continues until the user types `no` or `done`.

---

## System Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│   first_layer_monitoring_agent  │  ← Medical screening (LLM, JSON output)
│   Risk: GREEN / YELLOW / RED    │
└───────────────┬─────────────────┘
                │ GREEN only
                ▼
┌───────────────────────────────┐
│   Body + Activity + Diet      │  ← Structured survey collection
│   Survey Collection           │
└───────────┬───────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌──────────┐  ┌──────────────────┐
│ calorie_ │  │ nutrition_       │  ← Two independent LLM agents
│ counter  │  │ counter_agent    │
│ _agent   │  │ (intake kcal)    │
│(burn kcal│  └────────┬─────────┘
└────┬─────┘           │
     └────────┬─────────┘
              ▼
┌─────────────────────────────────────┐
│   nutrition_report_rag_agent        │
│   ┌─────────────────────────────┐   │
│   │  RAG Context (rag/ folder)  │   │  ← Knowledge retrieval
│   │  - Nutrition data           │   │
│   │  - Diet type guides         │   │
│   │  - Product/price info       │   │
│   └─────────────────────────────┘   │
│   → Generates Markdown report       │
└───────────────┬─────────────────────┘
                │
                ▼
       Revision Loop (user feedback)
                │
                ▼
    nutrition_reports/{username}_nutrition_report.md
```

**Key design decisions:**
- **Fail-closed risk management**: anything other than a clear GREEN stops the pipeline.
- **Separation of agents**: calorie burn and calorie intake are estimated independently to avoid cross-contamination of prompts.
- **RAG over fine-tuning**: the knowledge base can be updated without retraining — ideal for nutrition guidelines that evolve.
- **Audit trail**: every user record is saved to `df_user.csv` with pipeline status, risk level, timestamps, and the report path.

---

## Prompt Templates

### Agent 1 — `first_layer_monitoring_agent`

```
You are first_layer_monitoring_agent for an AI nutrition system.

HARD BLOCK (return RED):
Eating disorder, type 1 diabetes, uncontrolled type 2 diabetes, kidney disease,
liver failure, heart failure, recent heart attack/stroke, active cancer treatment,
pregnancy, breastfeeding, adolescent weight-loss request, severe underweight,
malnutrition, anaphylaxis.

MEDICAL REVIEW REQUIRED (return YELLOW):
Hypothyroidism, PCOS, insulin resistance, IBS, GERD, epilepsy, warfarin,
GLP-1 medications, steroids, antidepressants, diuretics.

SOFT WARNING (return GREEN_WITH_WARNING):
Vegetarian, vegan, intermittent fasting, keto, lactose intolerance, night shift.

Return only valid JSON: { "risk_level": "...", "reason": "...", "should_stop_pipeline": true/false }
```

**Prompt iterations:** Three versions were tested:
- v1 — Free-text risk assessment (inconsistent outputs, no JSON enforcement)
- v2 — JSON enforced but risk categories were ambiguous, YELLOW would sometimes continue
- v3 *(current)* — Explicit GREEN/YELLOW/RED with `should_stop_pipeline` flag; fail-closed logic moved to Python normalizer

### Agent 2 — `calorie_counter_agent`

```
Estimate approximate daily calories burned (TDEE).
Use: height, weight, sex, steps_per_day, lifestyle_assessment (0–10),
sport_count_per_week, sport_type.
Return only one integer. No explanation. No JSON.
```

### Agent 3 — `nutrition_counter_agent`

```
Estimate approximate daily calorie intake from food and drinks.
Use: meal descriptions, portion_size_assessment (0–10),
restaurant_or_fast_food_times_per_week, sweet_food_times_per_week.
Return only one integer. No explanation. No JSON.
```

### Agent 4 — `nutrition_report_rag_agent`

```
Create a practical nutrition report using:
1. User survey data
2. Estimated maintenance calories / calories burned
3. Estimated current daily calorie intake
4. User goal (weight_loss or muscle_gain)
5. Reference knowledge loaded from the rag/ folder

Safety rules: No diagnoses. No plans for RED-risk users.
Respect allergies and strongly disliked foods.
Avoid extreme deficits, crash diets, detoxes, or unsafe supplements.

Output: Markdown with four required sections:
  # Nutrition Report for {username}
  ## 1. Current State Analysis
  ## 2. Weekly Nutrition Plan (Monday–Sunday)
  ## 3. Daily Proportions
  ## 4. Notes and Safety Considerations
```

---

## RAG Knowledge Base

The `rag/` folder contains the retrieval-augmented knowledge used by the report agent:

| File | Contents |
|---|---|
| `Nutriation(1).txt` | Comprehensive nutrition reference (~440 KB) |
| `diet_types_nutrition_bridge(1).txt` | Mapping between diet types and macro targets |
| `diet_with_grams(1).txt` | Portion guidelines in grams |
| `product_names_prices_100g.txt` | Local product database with prices per 100g |
| `Diet types(2).txt` | Diet type descriptions |
| `Data Model(2).txt` | Internal data model documentation |
| `diverse_customer_profiles(4).txt` | Example user profiles for context |
| `medproblem(1).txt` | Medical condition reference for risk screening |

To add new knowledge, drop any `.txt`, `.md`, `.csv`, or `.json` file into `rag/`. No code changes needed.

---

## Data Storage

User records are persisted to `df_user.csv` with the following fields:

| Field | Description |
|---|---|
| `user_id` | UUID generated at registration |
| `username` | User-chosen unique identifier |
| `pipeline_status` | `COMPLETED`, `STOPPED`, or `FAILED` |
| `risk_level` | `GREEN`, `YELLOW`, or `RED` |
| `calories_burnt_estimated` | Agent 2 output (kcal/day) |
| `calories_consumed_estimated` | Agent 3 output (kcal/day) |
| `nutrition_report_path` | Path to the saved Markdown report |
| `created_at_utc` / `updated_at_utc` | ISO-8601 timestamps |

Generated reports are saved to `nutrition_reports/{username}_nutrition_report.md`.

---

## Known Limitations & Edge Cases

- **Calorie estimates are approximate** — LLM-based TDEE estimation is less precise than the Mifflin-St Jeor equation with proper activity multipliers. Treat outputs as rough guidance.
- **RAG is full-context, not vector search** — all files in `rag/` are concatenated and sent in the prompt. This works well at current scale but will hit token limits if the folder grows beyond ~150 KB.
- **No user authentication** — the current version identifies users by a self-chosen username with no authentication layer.
- **English and mixed-language inputs** — the system handles English and Russian inputs (seen in test data), but responses are generated in English. Non-Latin characters in usernames are sanitized for file paths.
- **Single-session only** — there is no mechanism to resume a partially completed pipeline session.
- **OpenAI dependency** — all agents use `gpt-4.1-mini`. Switching providers requires updating the client in each agent file.

---

## Ethical Considerations

- **Medical safety gate is mandatory** — no user can bypass the first-layer risk screening.
- **No diagnosis or treatment** — all reports include explicit disclaimers and recommend professional consultation.
- **Data minimization** — only data necessary for the nutrition plan is collected.
- **Transparency** — risk level and the agent's reason are stored and visible in `df_user.csv`.
- **Allergy respect** — allergies and strongly disliked foods are enforced in every report and revision.

---

## Team

| Member | Role |
|---|---|
| [Name 1] | Pipeline architecture, risk agent |
| [Name 2] | RAG design, report agent |
| [Name 3] | Calorie agents, evaluation |
| [Name 4] | Streamlit app, deployment |

---

## License

MIT License — see `LICENSE` for details.
