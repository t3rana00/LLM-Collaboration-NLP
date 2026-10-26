import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LEHMUS_API_KEY"),
    base_url=os.getenv("LEHMUS_BASE_URL")
)

response = client.chat.completions.create(
    model=os.getenv("LEHMUS_MODEL_ID"),
    messages=[
        {
            "role": "system",
            "content": "You are a Natural Language Inference classifier."
        },
        {
            "role": "user",
            "content": """
Premise: All students passed the examination.
Hypothesis: Some students failed the examination.

Classify as entailment, contradiction, or neutral.
Explain briefly.
"""
        }
    ],
    temperature=0,
    max_tokens=200
)

print(response.choices[0].message.content)