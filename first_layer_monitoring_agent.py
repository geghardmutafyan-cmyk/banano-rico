import os
from openai import OpenAI
import json
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are first_layer_monitoring_agent for an AI nutrition system.

Use this medical safety policy:

HARD BLOCK:
If the user likely has any serious contraindication such as eating disorder,
type 1 diabetes, uncontrolled type 2 diabetes, insulin-dependent diabetes,
kidney disease, liver failure, heart failure, recent heart attack/stroke,
active cancer treatment, pregnancy, breastfeeding, adolescent weight-loss request,
severe underweight, malnutrition, anaphylaxis, severe allergies, or unsafe goals,
return RED.

MEDICAL REVIEW REQUIRED:
If user mentions hypothyroidism, hyperthyroidism, PCOS, insulin resistance,
IBS, GERD, chronic gastritis, epilepsy, migraines, autoimmune disease,
depression, anxiety, OCD, ADHD, warfarin, GLP-1, steroids, antidepressants,
or diuretics, return YELLOW.

SOFT WARNING:
If user mentions vegetarian, vegan, intermittent fasting, keto, intolerance,
lactose intolerance, night shift, athletic training, return GREEN_WITH_WARNING.

Return only valid JSON:
{
  "risk_level": "RED | YELLOW | GREEN | GREEN_WITH_WARNING",
  "reason": "short reason",
  "should_stop_pipeline": true/false
}
"""


def run_first_layer_monitoring_agent(user_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    return json.loads(content)