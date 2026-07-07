from google import genai
import os
from build_prompt import build_prompt
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"))


def initialize_chat(report):

    prompt = build_prompt(report)

    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=prompt
    )
    return interaction.id


def chat(user_input, interaction_id=None):

    if interaction_id:
        interaction = client.interactions.create(
            model="gemini-3.5-flash",
            input=user_input,
            previous_interaction_id=interaction_id
        )
    else:
        interaction = client.interactions.create(
            model="gemini-3.5-flash",
            input=user_input
        )

    return {
        "response": interaction.output_text,
        "interaction_id": interaction.id
    }
