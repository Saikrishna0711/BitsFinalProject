import json, sys, itertools, os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def gpt_emotion(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=4,
        messages=[
            {"role": "system",
             "content": "Return exactly one word out of "
                        "[neutral, happy, sad, angry, disgust, fear]."},
            {"role": "user", "content": text}
        ],
    )
    return resp.choices[0].message.content.strip().lower()
