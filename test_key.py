import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY")

if key:
    print("API key found successfully!")
else:
    print("API key NOT found.")