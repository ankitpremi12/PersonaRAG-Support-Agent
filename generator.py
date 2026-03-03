import os
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

TONE_MAP = {
    "technical_expert": "Be precise and technical. Include config values and code snippets. Skip pleasantries.",
    "frustrated_user": "Be empathetic and calm. Acknowledge frustration. Use numbered steps. Plain language.",
    "business_executive": "Be concise and outcome-focused. No technical details. Max 5 sentences.",
}
VERBOSITY_MAP = {
    "technical_expert": "detailed and complete",
    "frustrated_user": "medium length, step-by-step",
    "business_executive": "brief, under 100 words",
}

def generate_response(message: str, persona: str, context_docs: list) -> dict:
    tone = TONE_MAP.get(persona, TONE_MAP["frustrated_user"])
    verbosity = VERBOSITY_MAP.get(persona, VERBOSITY_MAP["frustrated_user"])
    context_str = "\n\n".join(f"[{d['id']} - {d['title']}]\n{d['content']}" for d in context_docs)
    prompt = f"""You are a customer support agent for a SaaS company.
Persona: {persona} | Tone: {tone} | Length: {verbosity}
Rules: Use ONLY the KB context. Never invent features. Do NOT say "Certainly!" or "Great question!". End with "— Support Team".

KB Context:
{context_str}

Customer Message: {message}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return {
        "response_text": response.text,
        "model_used": "gemini-2.0-flash",
        "docs_used": [d["id"] for d in context_docs]
    }
