"""Deterministic no-LLM provider.

Two jobs:
1. Tests: the agent loop, guardrails and tools are exercised with zero model variance.
2. Degraded mode: if every LLM endpoint is down, shoppers still get grounded answers.

It plays the same protocol as a real model: decide a tool call, then answer from the result.
"""

from __future__ import annotations

import json
import re

from ..text.arabic import normalize, reply_language
from .base import ToolCall, Turn

_ORDER = re.compile(r"\bORD-\d{5}\b", re.IGNORECASE)
_LAST4 = re.compile(r"(?<!\d)\d{4}(?!\d)")
_HELP_WORDS = [
    "استرداد",
    "ارجع",
    "رجع",
    "فلوس",
    "دفع",
    "ادفع",
    "كارت",
    "محفظه",
    "رصيد",
    "باسورد",
    "كلمه السر",
    "حساب",
    "كوبون",
    "اهدي",
    "هديه",
    "كود",
    "حجز",
    "الغي",
    "الغاء",
    "خدمه العملاء",
    "اكلم",
    "رفض",
    "refund",
    "return",
    "pay",
    "payment",
    "card",
    "wallet",
    "password",
    "account",
    "voucher",
    "redeem",
    "gift",
    "promo",
    "code",
    "booking",
    "book",
    "cancel",
    "support",
    "contact",
    "refused",
]
_HUMAN_WORDS = [
    "عايز اكلم حد",
    "موظف",
    "بني ادم",
    "human",
    "agent",
    "real person",
    "شكوي",
    "complaint",
]


class ScriptedProvider:
    name = "scripted"
    model = "rules-v1"

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> Turn:
        last = history[-1]
        user = next(m["content"] for m in reversed(history) if m["role"] == "user")
        if last["role"] == "tool":
            return Turn(
                text=self._answer(last["name"], json.loads(last["content"]), user),
                provider=self.name,
                model=self.model,
            )
        call = self._route(user)
        if call is None:
            ar = reply_language(user) == "ar"
            msg = (
                "محتاج رقم الطلب (ORD-xxxxx) وآخر ٤ أرقام من موبايلك علشان أشوف الطلب."
                if ar
                else "Please share the order id (ORD-xxxxx) and the last 4 digits of your phone."
            )
            return Turn(text=msg, provider=self.name, model=self.model)
        return Turn(
            text="", tool_calls=[call], stop="tool_use", provider=self.name, model=self.model
        )

    def _route(self, user: str) -> ToolCall | None:
        n = normalize(user)
        if any(w in n for w in _HUMAN_WORDS):
            return ToolCall("s1", "escalate_to_human", {"reason": user[:200]})
        order = _ORDER.search(user)
        # explicit order references only: "I want to order food" is a search, not a lookup
        if order or any(w in n for w in ("طلبي", "my order", "order status", "اوردر")):
            rest = n.replace(order.group(0).lower(), " ") if order else n
            last4 = _LAST4.search(rest)
            if not (order and last4):
                return None
            return ToolCall(
                "s1",
                "order_status",
                {"order_id": order.group(0).upper(), "phone_last4": last4.group(0)},
            )
        if any(w in n for w in _HELP_WORDS):
            return ToolCall("s1", "search_help", {"query": user})
        return ToolCall("s1", "search_offers", {"query": user, "limit": 3})

    @staticmethod
    def _answer(tool: str, result: dict, user: str) -> str:
        ar = reply_language(user) == "ar"
        if "error" in result:
            return (
                "مش لاقي طلب بالبيانات دي، اتأكد من رقم الطلب وآخر ٤ أرقام."
                if ar
                else "I couldn't find an order with those details; please check the id and digits."
            )
        if tool == "search_offers":
            offers = result["offers"]
            if not offers:
                return "للأسف مفيش عروض مطابقة دلوقتي." if ar else "No matching offers right now."
            if ar:
                lines = [
                    f"- {o['title_ar']} من {o['merchant_ar']} ({o['area_ar']}): {o['price_egp']} جنيه، خصم {o['discount_pct']}٪ [{o['offer_id']}]"
                    for o in offers
                ]
                return "دي أحسن العروض اللي لقيتها:\n" + "\n".join(lines)
            lines = [
                f"- {o['title_en']} at {o['merchant_en']} ({o['area_en']}): {o['price_egp']} EGP, {o['discount_pct']}% off [{o['offer_id']}]"
                for o in offers
            ]
            return "Here are the best matches:\n" + "\n".join(lines)
        if tool == "search_help":
            arts = result["articles"]
            if not arts:
                return (
                    "مش لاقي إجابة، تحب أحولك لخدمة العملاء؟"
                    if ar
                    else "I couldn't find that; shall I connect you to support?"
                )
            a = arts[0]
            return f"{a['answer_ar'] if ar else a['answer_en']} [{a['article_id']}]"
        if tool == "order_status":
            o = result["offer"]
            return (
                f"طلبك {result['order_id']} ({o['title_ar']}) حالته: {result['status']}، صالح لحد {result['valid_until']}."
                if ar
                else f"Order {result['order_id']} ({o['title_en']}) is {result['status']}, valid until {result['valid_until']}."
            )
        if tool == "escalate_to_human":
            return (
                f"حولتك لموظف خدمة العملاء، رقم التذكرة {result['ticket']}."
                if ar
                else f"I've passed you to a human agent, ticket {result['ticket']}."
            )
        return json.dumps(result, ensure_ascii=False)
