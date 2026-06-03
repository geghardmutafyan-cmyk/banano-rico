import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are calorie_counter_agent for an AI nutrition pipeline.

Your task:
Estimate the user's approximate daily calories burnt / maintenance calories.

Use:
- height in cm
- weight in kg
- sex: M or F
- steps_per_day
- lifestyle_assessment from 0 to 10
- sport_count_per_week
- sport_type

Important:
- Return only one approximate integer number.
- Do not return explanation.
- Do not return JSON.
- If age is missing, assume age = 25.
- Estimate daily maintenance calories, also known as total daily energy expenditure.
"""


def run_calorie_counter_agent(user_data: dict) -> int:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_data, ensure_ascii=False),
            },
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content.strip()

    return int(content)