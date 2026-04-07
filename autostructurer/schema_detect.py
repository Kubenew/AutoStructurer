def detect_schema(text: str) -> str:
    t = text.lower()

    if "invoice" in t or "amount due" in t or "iban" in t:
        return "invoice"
    if "contract" in t or "agreement" in t or "signed by" in t:
        return "contract"
    if "meeting" in t or "agenda" in t or "minutes" in t:
        return "meeting_notes"
    if "error" in t or "exception" in t or "stack trace" in t:
        return "log"
    if "complaint" in t or "lawsuit" in t or "claim" in t:
        return "legal_complaint"

    return "generic_text"
