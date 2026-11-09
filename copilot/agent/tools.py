"""The copilot's tools: typed schemas (shared by every provider and the MCP server) + implementations.

Tool results are JSON strings. Prices and ids in replies must come from these results
(guardrails.ungrounded_prices checks it).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import cached_property

from ..data.catalog import Offer, load
from ..data.faq import ARTICLES
from ..retrieval.search import BM25, DenseIndex, Embedder, HybridRetriever

TOOL_SPECS: list[dict] = [
    {
        "name": "search_offers",
        "description": (
            "Search the Kanz deals catalogue. Use for any request to find, compare or price an "
            "offer (restaurants, gyms, spas, clinics, hotels, courses, repairs...). Pass the "
            "shopper's own words as query; Arabic, Egyptian dialect, Arabizi and English all work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What the shopper is looking for"},
                "max_price": {
                    "type": "integer",
                    "description": "Upper price limit in EGP, if given",
                },
                "limit": {"type": "integer", "description": "How many offers to return (1-5)"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_offer",
        "description": "Full details of one offer by its id (e.g. KZ-0042): price, validity, merchant, terms.",
        "input_schema": {
            "type": "object",
            "properties": {"offer_id": {"type": "string"}},
            "required": ["offer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_help",
        "description": (
            "Search the help centre for policies and how-to answers: refunds, redeeming vouchers, "
            "payments, bookings, account, wallet, gifts, promo codes, contacting support."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "order_status",
        "description": (
            "Look up an order. Requires the order id (ORD-xxxxx) AND the last 4 digits of the "
            "phone number on the account; ask the shopper for both before calling. Never guess them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "phone_last4": {"type": "string", "description": "Exactly 4 digits"},
            },
            "required": ["order_id", "phone_last4"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the conversation to a human agent. Use when the shopper asks for a person, is "
            "disputing a charge, reports a merchant problem the help centre cannot solve, or is upset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class Order:
    order_id: str
    phone_last4: str
    offer_id: str
    status: str
    purchased_on: str
    valid_until: str


def _orders(offers: list[Offer], n: int = 200, seed: int = 5) -> dict[str, Order]:
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        o = rng.choice(offers)
        bought = date(2026, 10, 1) + timedelta(days=rng.randrange(0, 30))
        oid = f"ORD-{10001 + i}"
        out[oid] = Order(
            oid,
            f"{rng.randrange(0, 10000):04d}",
            o.offer_id,
            rng.choice(["paid", "paid", "redeemed", "refunded", "expired"]),
            bought.isoformat(),
            o.valid_until,
        )
    return out


class Toolbox:
    def __init__(self, dense: bool | None = None):
        if dense is None:  # COPILOT_DENSE=0 -> lexical + slots only (tests, no embedding endpoint)
            dense = os.environ.get("COPILOT_DENSE", "1") != "0"
        self.offers = load()
        self.by_id = {o.offer_id: o for o in self.offers}
        self.orders = _orders(self.offers)
        self._dense = dense
        self.escalations: list[str] = []

    @cached_property
    def retriever(self) -> HybridRetriever:
        ids, docs = [o.offer_id for o in self.offers], [o.document() for o in self.offers]
        facets = {
            o.offer_id: {
                "category": o.category,
                "area": o.area_en,
                "city": o.city_en,
                "price": o.price,
            }
            for o in self.offers
        }
        dense = None
        if self._dense:
            try:
                dense = DenseIndex(ids, docs, Embedder())
            except Exception:  # embedding endpoint down: degrade to lexical + slots, never fail
                dense = None
        return HybridRetriever(BM25(ids, docs), dense, facets=facets)

    @cached_property
    def help_index(self) -> BM25:
        return BM25([a.article_id for a in ARTICLES], [a.document() for a in ARTICLES])

    # ------------------------------------------------------------------ tools ---
    def search_offers(self, query: str, max_price: int | None = None, limit: int = 3) -> dict:
        limit = max(1, min(int(limit or 3), 5))
        q = f"{query} تحت {max_price}" if max_price else query
        hits = self.retriever.search(q, limit)
        offers = [self.by_id[h.doc_id] for h in hits]
        if max_price:
            offers = [o for o in offers if o.price <= max_price]
        return {"offers": [_card(o) for o in offers], "count": len(offers)}

    def get_offer(self, offer_id: str) -> dict:
        o = self.by_id.get(offer_id.strip().upper())
        if not o:
            return {"error": f"no offer with id {offer_id}"}
        return _card(o) | {
            "original_price": o.original_price,
            "city": o.city_en,
            "terms": _terms(o),
        }

    def search_help(self, query: str) -> dict:
        by_id = {a.article_id: a for a in ARTICLES}
        hits = self.help_index.search(query, 2)
        return {
            "articles": [
                {
                    "article_id": h.doc_id,
                    "question_ar": by_id[h.doc_id].question_ar,
                    "answer_ar": by_id[h.doc_id].answer_ar,
                    "answer_en": by_id[h.doc_id].answer_en,
                }
                for h in hits
            ]
        }

    def order_status(self, order_id: str, phone_last4: str) -> dict:
        order = self.orders.get(order_id.strip().upper())
        # Same message for "no such order" and "wrong phone": never confirm an order exists.
        if not order or order.phone_last4 != str(phone_last4).strip():
            return {"error": "order not found or verification failed"}
        offer = self.by_id[order.offer_id]
        return asdict(order) | {"offer": _card(offer)}

    def escalate_to_human(self, reason: str) -> dict:
        ticket = f"TCK-{len(self.escalations) + 1:05d}"
        self.escalations.append(reason)
        return {"ticket": ticket, "message": "A human agent will reply within 2 hours (9am-11pm)."}

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if name not in {t["name"] for t in TOOL_SPECS} or fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(fn(**args), ensure_ascii=False)
        except TypeError as exc:  # bad/missing arguments from the model
            return json.dumps({"error": f"invalid arguments for {name}: {exc}"})


def _card(o: Offer) -> dict:
    return {
        "offer_id": o.offer_id,
        "title_ar": o.title_ar,
        "title_en": o.title_en,
        "merchant_ar": o.merchant_ar,
        "merchant_en": o.merchant_en,
        "area_ar": o.area_ar,
        "area_en": o.area_en,
        "category": o.category,
        "price_egp": o.price,
        "discount_pct": o.discount_pct,
        "valid_until": o.valid_until,
    }


def _terms(o: Offer) -> str:
    booking = (
        "Booking required."
        if o.category in {"spa", "clinic", "dental", "hotel"}
        else "No booking needed."
    )
    return f"Valid at {o.merchant_en}, {o.area_en} branch only. {booking} One voucher per visit."
