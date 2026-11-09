"""Guardrails that run outside the model, so they hold whatever the model does.

redact_pii        - logs and traces never contain phone numbers, emails, card or national ids
detect_injection  - user text or tool output that tries to re-instruct the assistant
ungrounded_prices - every price in a reply must have appeared in this turn's tool results
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..text.arabic import normalize

# Order matters: the most specific patterns run first (a national id is also 14 digits,
# which the generic card pattern would otherwise claim).
_PII = [
    ("phone", re.compile(r"(?<!\d)(?:\+?20|0)1[0125]\d{8}(?!\d)")),
    ("national_id", re.compile(r"(?<!\d)[23]\d{13}(?!\d)")),  # Egyptian national id: 14 digits
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
]


def redact_pii(text: str) -> str:
    text = normalize_digits(text)
    for label, rx in _PII:
        text = rx.sub(f"[{label}]", text)
    return text


def normalize_digits(text: str) -> str:
    return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


# Phrases that try to override instructions or exfiltrate the prompt, in Arabic and English.
_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)",
    r"disregard (your|the) (instructions|system prompt|rules)",
    r"(reveal|show|print|repeat) (your|the) (system prompt|instructions|hidden prompt)",
    r"you are now (?!looking)",  # role hijack ("you are now DAN")
    r"act as (an? )?(admin|developer|system)",
    r"تجاهل (كل |جميع )?(التعليمات|الاوامر)",
    r"انس(ي|ى)? (كل )?(التعليمات|اللي فات)",
    r"(اعرض|اكتب|قول(ي)?) (لي )?(التعليمات|البرومبت|السيستم)",
    r"انت (دلوقتي|الان) (مش|مبقتش)",
]
_INJECTION_RX = re.compile("|".join(_INJECTION), re.IGNORECASE)


def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RX.search(normalize(text)))


_PRICE = re.compile(r"(\d[\d,]*)\s*(?:جنيه|ج\.?م|egp|le|pounds?)", re.IGNORECASE)


def prices_in(text: str) -> set[int]:
    return {
        int(m.group(1).replace(",", "")) for m in _PRICE.finditer(normalize_digits(text).lower())
    }


def ungrounded_prices(reply: str, tool_outputs: list[str]) -> set[int]:
    """Prices quoted in the reply that no tool returned this turn (a hallucination signal)."""
    seen: set[int] = set()
    for out in tool_outputs:
        seen |= _price_values(out)
    return {p for p in prices_in(reply) if p not in seen}


def _price_values(tool_output: str) -> set[int]:
    """Only real price fields count as evidence; ids like KZ-0001 must not 'ground' a price of 1."""
    try:
        data = json.loads(tool_output)
    except (json.JSONDecodeError, TypeError):
        return set()
    found: set[int] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if "price" in k and isinstance(v, int | float):
                    found.add(int(v))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return found


@dataclass
class GuardReport:
    injection_in_input: bool = False
    injection_in_tools: list[str] = field(default_factory=list)
    ungrounded: list[int] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.injection_in_input or self.injection_in_tools or self.ungrounded)
