import json
import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are nutrition_counter_agent for an AI nutrition pipeline.

Your task:
Estimate the user's approximate daily calorie intake from food and drinks.

Use only intake-related data:
- meals_per_day
- snacks_per_day
- breakfast_description
- lunch_description
- dinner_description
- snack_description
- drinks_description
- portion_size_assessment from 0 to 10
- restaurant_or_fast_food_times_per_week
- sweet_food_times_per_week

Behavioral preference data may be present, such as loved_foods and hated_foods.
Do not use loved_foods or hated_foods for this calorie estimate.
That preference data is collected only for future diet personalization.

Important:
- Return only one approximate integer number.
- Do not return explanation.
- Do not return JSON.
- Estimate approximate daily calories consumed, not calories burned.
- If data is incomplete, make a reasonable estimate from the available survey answers.
"""


def _extract_first_integer(text: str) -> int:
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"nutrition_counter_agent returned no integer: {text!r}")
    return int(match.group())


def run_nutrition_counter_agent(user_data: dict) -> int:
    """Return approximate daily calorie intake as an integer kcal value."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_data, ensure_ascii=False),
            },
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    return _extract_first_integer(content)
