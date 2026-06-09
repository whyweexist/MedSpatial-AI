"""Clinical language safety rules."""

CERTAINTY_PHRASES = ("definitely has", "confirmed diagnosis", "certainly has", "proves that")


def validate_clinical_language(text: str) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in CERTAINTY_PHRASES):
        raise ValueError("Clinical answer expresses unsupported diagnostic certainty")
    return text
