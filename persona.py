import os
import json
import logging
from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. Create a .env file with:\n  GEMINI_API_KEY=your_key_here\n"
        "Get your key at: https://aistudio.google.com/app/apikey"
    )

client = genai.Client(api_key=_api_key)

PERSONA_PROMPT = """
You are a customer support analyst. Classify the customer message into exactly one persona.
Personas:
- technical_expert: Uses technical jargon, mentions logs/APIs/config/errors
- frustrated_user: Emotional language, caps, exclamation marks, repeated failures
- business_executive: Outcome/ROI focused, mentions stakeholders, business language

Return ONLY valid JSON, no markdown, no explanation:
{{"persona": "technical_expert | frustrated_user | business_executive", "confidence": 0.0, "signals": []}}

Customer Message:
{message}
"""

def detect_persona(message: str) -> dict:
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=PERSONA_PROMPT.format(message=message)
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        valid_personas = {"technical_expert", "frustrated_user", "business_executive"}
        if result.get("persona") not in valid_personas:
            result["persona"] = "frustrated_user"
            result["confidence"] = 0.4
        return result
    except Exception as e:
        logger.error(f"Persona detection failed: {e}")
        return {"persona": "frustrated_user", "confidence": 0.3, "signals": ["fallback"]}
