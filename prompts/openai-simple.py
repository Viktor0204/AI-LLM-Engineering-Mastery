from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the purpose of life?"}
    ]
)

print(response.choices[0].message.content)