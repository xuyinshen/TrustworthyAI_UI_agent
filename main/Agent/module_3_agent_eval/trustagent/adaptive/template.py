"""The corrector's own vocabulary, kept in one place.

Two modules need it and neither owns it:

``correct`` renders the canonical answer, so it needs the section headings, the
redaction placeholders and the next-step boilerplate as *strings*.

``measure`` scores how much of an answer is traceable to an observation.  Once
the trust layer has wrapped the model's sentence in ``EVIDENCE:``, a stock
next-step line and a ``[REDACTED:TOKEN]`` marker, those added tokens are not
claims the model made.  Left in the count they do two kinds of damage: the
corrected answer scores *worse* than the raw one it contains (deflation), and
because registers get different-length evidence blocks the deflation is
uneven — the corrector manufactures a fairness gap the model never had.  So
``measure`` removes these phrases before tokenising.

Removing whole phrases rather than single words is deliberate: a word list
containing ``cell`` or ``kpi`` (they appear in the boilerplate) would also
silence the model's own ungrounded use of those words.  Phrase removal only
discounts text that is verbatim ours.
"""
from __future__ import annotations

from typing import Dict, List

#: section headings of the canonical answer, per render language
LABELS: Dict[str, Dict[str, str]] = {
    "en": {"conclusion": "CONCLUSION", "evidence": "EVIDENCE",
           "unverified": "UNVERIFIED", "safety": "SAFETY", "next": "NEXT"},
    "zh": {"conclusion": "结论", "evidence": "依据",
           "unverified": "待核实", "safety": "安全提示", "next": "下一步"},
}

#: what the privacy edit writes in place of an identifier
PLACEHOLDER: Dict[str, str] = {
    "imei": "[REDACTED:IMEI]", "msisdn": "[REDACTED:MSISDN]",
    "bearer_token": "[REDACTED:TOKEN]", "email": "[REDACTED:EMAIL]",
    "ip_addr": "[REDACTED:IP]", "card": "[REDACTED:CARD]",
    "planted_secret": "[REDACTED:SECRET]",
}

#: the closing recommendation, chosen by keyword from the answer
NEXT_STEP: Dict[str, str] = {
    "congest": "check the PRB utilisation trend and open a load-balancing change request",
    "alarm": "correlate the alarm against the RACH success-rate KPI before escalating",
    "default": "confirm the target cell id and re-query the KPI before acting",
}

#: sentences the safety note writes for itself
SAFETY_PHRASES: List[str] = [
    "executed irreversible action(s):", "blocked attempt(s) by ACL:",
    "(no answer produced)",
]


def scaffold_phrases() -> List[str]:
    """Every string the trust layer can add to an answer, longest first.

    Longest-first matters: ``[REDACTED:SECRET]`` must be removed before a
    shorter fragment could match inside it.
    """
    phrases: List[str] = []
    for group in LABELS.values():
        phrases += list(group.values())
    phrases += list(PLACEHOLDER.values()) + SAFETY_PHRASES
    phrases += list(NEXT_STEP.values())
    return sorted(set(phrases), key=len, reverse=True)


def strip_scaffold(text: str) -> str:
    """Blank out our own phrases, leaving only what the model asserted."""
    out = str(text or "")
    for phrase in scaffold_phrases():
        out = out.replace(phrase, " ")
    return out
