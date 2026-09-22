"""
privacy/redactor.py
───────────────────
Single-file PII + sensitive financial data redaction.

Two layers, always both run:

  INPUT
    │
    ▼
  ┌─────────────────────────────────────────────────┐
  │  Layer 1 — Presidio NLP  (lazy, optional)       │
  │  Catches: names, emails, phones, SSNs,          │
  │  credit cards, IBANs, bank numbers, IPs,        │
  │  passports, driver's licences                   │
  │  Requires: presidio-analyzer + spaCy model      │
  │  Falls back silently if not installed           │
  └──────────────────────────┬──────────────────────┘
                             │
    ▼
  ┌─────────────────────────────────────────────────┐
  │  Layer 2 — Regex  (always active, zero deps)    │
  │  Catches: EINs, ABA routing numbers, IBANs,     │
  │  account refs, cards, SSNs, emails, phones,     │
  │  IPs, passports                                 │
  └──────────────────────────┬──────────────────────┘
                             │
    ▼
  REDACTED OUTPUT + findings list

Why two layers?
  Presidio is NLP-based — it excels at names, free-text entities.
  Regex is pattern-based — it's the only reliable way to catch
  financial IDs like EIN (12-3456789) or ABA routing (021000021)
  that have no semantic meaning for an NLP model.
"""
from __future__ import annotations

import re
import logging
import time
from typing import Optional

from .audit import log_redaction

logger = logging.getLogger(__name__)

# ── Entity → placeholder map ──────────────────────────────────────────────────
_ENTITY_PLACEHOLDERS: dict[str, str] = {
    "PERSON":            "[PERSON]",
    "EMAIL_ADDRESS":     "[EMAIL]",
    "PHONE_NUMBER":      "[PHONE]",
    "US_SSN":            "[SSN]",
    "US_DRIVER_LICENSE": "[DL]",
    "US_PASSPORT":       "[PASSPORT]",
    "CREDIT_CARD":       "[CARD]",
    "US_BANK_NUMBER":    "[BANK_ACCOUNT]",
    "IBAN_CODE":         "[IBAN]",
    "URL":               "[URL]",
    "IP_ADDRESS":        "[IP]",
    # DATE_TIME intentionally excluded — fiscal periods must not be redacted.
    # LOCATION intentionally excluded — country/state names are legitimate in filings.
}

_PRESIDIO_ENTITIES = list(_ENTITY_PLACEHOLDERS.keys())

# ── Regex patterns ─────────────────────────────────────────────────────────────
# Each: (label, compiled pattern, placeholder)
_REGEX_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("SSN",         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                              "[SSN]"),
    ("EIN",         re.compile(r"\b\d{2}-\d{7}\b"),                                    "[EIN]"),
    ("ABA_ROUTING", re.compile(r"\b(?:0[0-9]|1[0-2]|2[1-9]|3[0-2])\d{7}\b"),         "[ROUTING]"),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b"),                              "[CARD]"),
    ("IBAN",        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),                   "[IBAN]"),
    ("EMAIL",       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    ("PHONE",       re.compile(r"\b(?:\+1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"), "[PHONE]"),
    ("IPv4",        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),                        "[IP]"),
    ("PASSPORT",    re.compile(r"\b[A-Z]\d{8}\b"),                                     "[PASSPORT]"),
    ("ACCOUNT_REF", re.compile(r"\bAcct(?:ount)?\s*[:#]?\s*\d[\d\- ]{6,}\b", re.I),  "[ACCOUNT]"),
]

# ── Presidio singleton (lazy) ─────────────────────────────────────────────────
_analyzer: Optional[object]   = None
_anonymizer: Optional[object] = None
_presidio_ready: bool         = False
_presidio_tried: bool         = False

_SPACY_MODELS = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]


def _init_presidio() -> bool:
    """Try to initialise Presidio once. Subsequent calls are no-ops."""
    global _analyzer, _anonymizer, _presidio_ready, _presidio_tried
    if _presidio_tried:
        return _presidio_ready
    _presidio_tried = True

    try:
        import spacy                                                         # type: ignore
        from presidio_analyzer import AnalyzerEngine                        # type: ignore
        from presidio_analyzer.nlp_engine import NlpEngineProvider          # type: ignore
        from presidio_anonymizer import AnonymizerEngine                    # type: ignore

        # Pick the best installed spaCy model
        model = next((m for m in _SPACY_MODELS if spacy.util.is_package(m)), None)
        if model is None:
            # Try downloading the smallest one automatically
            import subprocess, sys
            logger.info("[Presidio] No spaCy model found — downloading en_core_web_sm …")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True, capture_output=True, timeout=300,
            )
            model = "en_core_web_sm"

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
        })
        _analyzer   = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        _anonymizer = AnonymizerEngine()
        _presidio_ready = True
        logger.info(f"[Presidio] Ready — using spaCy model: {model}")

    except Exception as exc:
        logger.warning(f"[Presidio] Not available ({exc}) — regex-only mode active.")
        _presidio_ready = False

    return _presidio_ready


def presidio_available() -> bool:
    """Return True if Presidio has successfully initialised."""
    return _presidio_ready


def active_engine_label() -> str:
    """Human-readable label of the NLP engine in use."""
    return "PRESIDIO" if _presidio_ready else "REGEX-ONLY"


# ── Core redact function ───────────────────────────────────────────────────────

def redact(text: str, source: str = "unknown") -> tuple[str, list[dict]]:
    """
    Run both redaction layers on *text*.

    Parameters
    ----------
    text   : Raw input string.
    source : Audit label ('user_query' | 'chunk').

    Returns
    -------
    (redacted_text, findings)
    findings = [{"entity_type": str, "count": int, "method": str}, …]
    """
    if not text or not text.strip():
        return text, []

    _init_presidio()   # no-op after first call
    findings: list[dict] = []
    working = text

    # ── Layer 1: Presidio NLP ─────────────────────────────────────────────────
    if _presidio_ready:
        try:
            from presidio_anonymizer.entities import OperatorConfig          # type: ignore

            results = _analyzer.analyze(text=working, entities=_PRESIDIO_ENTITIES, language="en")
            if results:
                operators = {
                    r.entity_type: OperatorConfig(
                        "replace",
                        {"new_value": _ENTITY_PLACEHOLDERS.get(r.entity_type, f"[{r.entity_type}]")},
                    )
                    for r in results
                }
                counts: dict[str, int] = {}
                for r in results:
                    counts[r.entity_type] = counts.get(r.entity_type, 0) + 1

                working = _anonymizer.anonymize(
                    text=working, analyzer_results=results, operators=operators
                ).text

                for etype, cnt in counts.items():
                    log_redaction(source=source, entity_type=etype,
                                  count=cnt, engine="presidio", context_snippet=text[:80])
                    findings.append({"entity_type": etype, "count": cnt, "method": "presidio"})

        except Exception as exc:
            logger.warning(f"[Presidio] Analysis error: {exc}")

    # ── Layer 2: Regex (always) ───────────────────────────────────────────────
    for label, pattern, placeholder in _REGEX_PATTERNS:
        matches = pattern.findall(working)
        if matches:
            working = pattern.sub(placeholder, working)
            log_redaction(source=source, entity_type=label,
                          count=len(matches), engine="regex", context_snippet=text[:80])
            findings.append({"entity_type": label, "count": len(matches), "method": "regex"})

    return working, findings
