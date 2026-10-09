
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("API key not found. Check your .env file.")

client = genai.Client(api_key=api_key)

try:
    print("Checking available models...\n")

    for model in client.models.list():
        actions = model.supported_actions or []

        if "generateContent" in actions:
            name = model.name or ""

            if "gemma" in name.lower():
                print(f"Gemma model: {name}")
                print(f"Supported actions: {actions}\n")

    print("Model listing completed.")

except Exception as error:
    print("Could not list models.")
    print(type(error).__name__ + ":", error)