import os
import logging
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

from persona import detect_persona
from retriever import retrieve
from generator import generate_response
from escalation import check_escalation_triggers, build_escalation_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Persona-Adaptive Customer Support Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    attempt_count: int = Field(default=1, ge=1)

class PersonaInfo(BaseModel):
    detected: str
    confidence: float
    signals: List[str]

class ChatResponse(BaseModel):
    response: str
    persona: PersonaInfo
    escalated: bool
    escalation_context: Optional[dict] = None
    docs_used: Optional[List[str]] = None
    session_id: Optional[str] = None

@app.get("/")
def root():
    return {"service": "Persona-Adaptive Customer Support Agent", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        persona_result = detect_persona(request.message)
    except Exception as e:
        logger.error(f"Persona detection error: {e}")
        raise HTTPException(status_code=500, detail="Persona detection failed.")

    persona = persona_result.get("persona", "frustrated_user")
    confidence = float(persona_result.get("confidence", 0.5))
    signals = persona_result.get("signals", [])

    try:
        retrieved_docs = retrieve(request.message, persona=persona, top_k=3)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        retrieved_docs = []

    escalation_check = check_escalation_triggers(
        message=request.message, persona=persona,
        confidence=confidence, attempt_count=request.attempt_count
    )

    if escalation_check["should_escalate"]:
        escalation_pkg = build_escalation_context(
            message=request.message, persona=persona, confidence=confidence,
            escalation_reasons=escalation_check["reasons"],
            retrieved_docs=retrieved_docs, attempt_count=request.attempt_count
        )
        return ChatResponse(
            response="I'm connecting you with a human support specialist right now. Please hold on.",
            persona=PersonaInfo(detected=persona, confidence=confidence, signals=signals),
            escalated=True,
            escalation_context=escalation_pkg["escalation_context"],
            docs_used=[d["id"] for d in retrieved_docs],
            session_id=request.session_id
        )

    try:
        generation_result = generate_response(
            message=request.message, persona=persona, context_docs=retrieved_docs
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Response generation failed.")

    return ChatResponse(
        response=generation_result["response_text"],
        persona=PersonaInfo(detected=persona, confidence=confidence, signals=signals),
        escalated=False,
        docs_used=generation_result["docs_used"],
        session_id=request.session_id
    )
