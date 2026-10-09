
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: API key not found.")

client = genai.Client(api_key=api_key)

MODEL = "models/gemma-4-26b-a4b-it"

prompt = """
You are the report-analysis component of RescueFlow AI.

Analyse this simulated incident report:
"Water is collecting near the main gate."

Return:
1. Incident type
2. Reported location
3. Reported facts
4. Missing information
5. One useful clarification question

Do not invent facts.
"""

try:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    print("\nSUCCESS: Gemma responded!")
    print("Model:", MODEL)
    print("\nRescueFlow AI analysis:\n")
    print(response.text)

except Exception as error:
    print("\nMODEL TEST FAILED")
    print(type(error).__name__ + ":", error)