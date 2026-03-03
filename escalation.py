"""
escalation.py - Determines if a conversation should be escalated to a human agent.
Packages full context for handoff.
"""

import logging

logger = logging.getLogger(__name__)

# Keywords that trigger automatic escalation
ESCALATION_KEYWORDS = [
    "human", "agent", "real person", "supervisor", "manager",
    "lawyer", "legal", "sue", "lawsuit", "fraud", "scam",
    "cancel account", "close account", "refund immediately",
]

FRUSTRATION_PHRASES = [
    "not working", "never works", "still broken", "this is ridiculous",
    "unacceptable", "worst", "terrible", "useless", "hate this",
]

CONFIDENCE_THRESHOLD = 0.45


def check_escalation_triggers(message: str, persona: str, confidence: float, attempt_count: int) -> dict:
    """
    Check multiple escalation conditions.

    Returns:
        dict with keys: should_escalate, reasons (list)
    """
    reasons = []
    message_lower = message.lower()

    # 1. Explicit request for human
    for kw in ESCALATION_KEYWORDS:
        if kw in message_lower:
            reasons.append(f"explicit_keyword: '{kw}'")
            break

    # 2. Low classification confidence
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"low_confidence: {confidence:.2f}")

    # 3. Highly frustrated persona + frustration phrase
    if persona == "frustrated_user":
        for phrase in FRUSTRATION_PHRASES:
            if phrase in message_lower:
                reasons.append(f"frustration_phrase: '{phrase}'")
                break

    # 4. Too many attempts
    if attempt_count >= 3:
        reasons.append(f"max_attempts_reached: {attempt_count}")

    should_escalate = len(reasons) > 0

    if should_escalate:
        logger.warning(f"Escalation triggered. Reasons: {reasons}")
    else:
        logger.info("No escalation needed.")

    return {
        "should_escalate": should_escalate,
        "reasons": reasons
    }


def build_escalation_context(
    message: str,
    persona: str,
    confidence: float,
    escalation_reasons: list[str],
    retrieved_docs: list[dict],
    attempt_count: int
) -> dict:
    """
    Package full context for human agent handoff.
    """
    return {
        "escalation_context": {
            "original_message": message,
            "detected_persona": persona,
            "persona_confidence": confidence,
            "escalation_reasons": escalation_reasons,
            "retrieved_docs": [d["id"] for d in retrieved_docs] if retrieved_docs else [],
            "attempt_count": attempt_count,
            "recommended_action": _recommend_action(persona, escalation_reasons)
        }
    }


def _recommend_action(persona: str, reasons: list[str]) -> str:
    if any("legal" in r or "fraud" in r or "lawsuit" in r for r in reasons):
        return "URGENT: Route to legal/compliance team immediately."
    if persona == "business_executive":
        return "Route to Senior Account Manager or Customer Success team."
    if persona == "frustrated_user":
        return "Route to empathy-trained support specialist. Customer may need retention offer."
    if persona == "technical_expert":
        return "Route to Tier 2 technical support."
    return "Route to general support queue."
