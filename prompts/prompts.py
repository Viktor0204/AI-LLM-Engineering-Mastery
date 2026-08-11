from anyio.itertools import Chain
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

model="gpt-4o-mini"

completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a travel blogger."},
        {"role": "user",
         "content": "Write a 500-word blog post about your recent trip to Paris. Make sure to give a step-by-step itinerary of your trip."}
    ],
    temperature=0.9, # randomness
    stream=True,
    # top_p=0.9,      # diversity
)
# print(completion.choices[0].message.content)
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
    print("\n")

# Temperature and top-p sampling
# completion = client.chat.completions.create(
#     model=model,
#     messages = [
#         {"role": "system", "content": "You are a creative writer."},
#         {"role": "user", "content": "Write a creative tagline for a coffee shop."}
#     ],
#     temperature=0.5, # randomness
#     # top_p=0.9,       # diversity
# )


# Open-ended prompts
# messages = [
#     {"role": "system", "content": "You are a philosopher."},
#     {"role": "user", "content": "What is the meaning of life?"}
# ]

# Role-playing prompts
# messages = [
#     {"role": "system", "content": "You are a character in a fantasy novel."},
#     {"role": "user", "content": "Describe the setting of the story."}
# ]


# Instructional prompts
# messages = [
#     {
#       "role": "system",
#       "content": "You are a knowledgeable personal trainer and writer."
#     },
#     {
#       "role": "user",
#       "content": "Write a 300-word summary of the benefits of exercise using bullet points."
#     }
# ]


# Chain of thought
# messages = [
#     {"role": "system", "content": "You are a math tutor."},
#     {"role": "user", "content": "Solve this math problem step by step: If John has 5 apples and gives 2 to Mary, how many does he have left?"}
# ]

# Zero-shot prompting
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"}
# ]


# Few-shots prompting
# messages = [
#     {"role": "system", "content": "You are a translator"},
#     {
#         "role": "user", "content": """Translate these sentences:
#         'Hello' -> 'Hola'
#         'Goodbye' -> 'Adiós'.
#         '.
#         Now translate: 'Thank you'.
#         """
#     }
# ]


# Simple prompting
# messages = [
#     {
#         "role": "system",
#         "content": "You are a eastern poet."
#     },
#     {
#         "role": "user",
#         "content": """write me a short poem about the moon.
#             Write the poem in the style of a haiku.
#             Make sure include a title for the poem."""
#     }
# ]