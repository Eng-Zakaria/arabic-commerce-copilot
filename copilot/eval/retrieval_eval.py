"""Retrieval eval: does the right offer come back, however the shopper types it?

Two splits, reported separately:
  templated   - generated from the catalogue in 6 variants (MSA, Egyptian dialect, Arabizi,
                English, typo-noised, city-level). Large, but shares vocabulary with the index.
  handwritten - natural phrasings written independently of the lexicons (filler, odd order,
                slang). Small, and the honest number: it is not derived from the index.

Relevance is defined by intent (category + area/city), never by listing offer ids, so it
stays correct if the catalogue is regenerated.

    uv run python -m copilot.eval.retrieval_eval            # all modes, writes artifacts/eval/
    uv run python -m copilot.eval.retrieval_eval --no-dense # skip embeddings (fast, CI)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..data.catalog import CATEGORIES, CITIES, Offer, load
from ..retrieval.search import BM25, DenseIndex, Embedder, HybridRetriever
from ..text.arabic import plain_tokenize

ARABIZI_CAT = {
    "restaurant": "mat3am",
    "cafe": "kafeh",
    "gym": "gym",
    "spa": "spa",
    "salon": "kwafeer",
    "clinic": "dktor",
    "dental": "dktor asnan",
    "kids": "malahy atfal",
    "course": "kors",
    "hotel": "fondo2",
    "electronics": "ta9le7 mobayl",
    "carwash": "8aseel 3arabeya",
}
DIALECT_FRAMES = [
    "عايز {c} في {a}",
    "فين احسن {c} ف {a}",
    "في {c} حلو {a}؟",
    "محتاج {c} قريب من {a}",
]
ARABIZI_FRAMES = ["3ayez {c} fe {a}", "feen {c} {a}", "{c} fe {a} ar5as 7aga"]
ENGLISH_FRAMES = ["{c} in {a}", "best {c} near {a}", "cheap {c} {a}"]


@dataclass(frozen=True)
class Query:
    text: str
    variant: str
    category: str | None
    area: str | None = None
    city: str | None = None


def relevant(q: Query, offers: list[Offer]) -> set[str]:
    return {
        o.offer_id
        for o in offers
        if (q.category is None or o.category == q.category)
        and (q.area is None or o.area_en == q.area)
        and (q.city is None or o.city_en == q.city)
    }


def _noise(text: str, rng: random.Random) -> str:
    """Realistic typing noise: hamza dropped, ta marbuta as ha, ya/alef maqsura swapped, tatweel."""
    swaps = [("أ", "ا"), ("إ", "ا"), ("ة", "ه"), ("ي", "ى"), ("ى", "ي")]
    for a, b in swaps:
        if rng.random() < 0.8:
            text = text.replace(a, b)
    if rng.random() < 0.5 and len(text) > 4:
        i = rng.randrange(1, len(text) - 1)
        text = text[:i] + "ـ" + text[i:]
    return text


def templated_queries(offers: list[Offer], seed: int = 11) -> list[Query]:
    rng = random.Random(seed)
    cat_meta = {k: (ar, en, syns) for k, ar, en, syns in CATEGORIES}
    pairs = sorted({(o.category, o.area_en) for o in offers})
    area_meta = {en: (ar, rz, city) for city_ar, city, al in CITIES for ar, en, rz in al}
    city_ar_of = {en: ar for ar, en, _ in CITIES}
    qs: list[Query] = []
    for cat, area in pairs:
        ar, en, syns = cat_meta[cat]
        area_ar, area_rz, _city = area_meta[area]
        qs += [
            Query(f"{ar} في {area_ar}", "msa", cat, area),
            Query(
                rng.choice(DIALECT_FRAMES).format(c=rng.choice(syns), a=area_ar),
                "dialect",
                cat,
                area,
            ),
            Query(
                rng.choice(ARABIZI_FRAMES).format(c=ARABIZI_CAT[cat], a=area_rz),
                "arabizi",
                cat,
                area,
            ),
            Query(rng.choice(ENGLISH_FRAMES).format(c=en, a=area), "english", cat, area),
            Query(_noise(f"{rng.choice([ar, *syns])} في {area_ar}", rng), "typo", cat, area),
        ]
    for cat in cat_meta:
        for _, city_en, _ in CITIES:
            if any(o.category == cat and o.city_en == city_en for o in offers):
                qs.append(
                    Query(
                        f"{rng.choice(cat_meta[cat][2])} في {city_ar_of[city_en]}",
                        "city",
                        cat,
                        city=city_en,
                    )
                )
    return qs


# Written independently of the lexicons: the way people actually type in chat.
HANDWRITTEN: list[Query] = [
    Query(
        "لو سمحت عايزة أعمل شعري عند كوافير حلو في المهندسين", "handwritten", "salon", "Mohandessin"
    ),
    Query(
        "ابني عيد ميلاده الأسبوع الجاي، فين أعمله حفلة في أكتوبر؟",
        "handwritten",
        "kids",
        "6th of October",
    ),
    Query(
        "شاشة الموبايل اتكسرت، في حد بيصلح في مدينة نصر؟", "handwritten", "electronics", "Nasr City"
    ),
    Query("نفسي في مساج بعد الشغل، حاجة في المعادي", "handwritten", "spa", "Maadi"),
    Query("هنتعشى برا النهارده في سموحة، اقترح مطعم", "handwritten", "restaurant", "Smouha"),
    Query("عربيتي محتاجة غسلة جامدة، التجمع", "handwritten", "carwash", "New Cairo"),
    Query("عايز أبدأ أتمرن، اشتراك جيم في زايد بكام؟", "handwritten", "gym", "Sheikh Zayed"),
    Query("سناني محتاجة تنضيف، دكتور سنان كويس في الدقي", "handwritten", "dental", "Dokki"),
    Query(
        "i want to learn english, any course in heliopolis?", "handwritten", "course", "Heliopolis"
    ),
    Query(
        "looking for a hotel night in alex for me and my wife",
        "handwritten",
        "hotel",
        None,
        "Alexandria",
    ),
    Query("7ad y3raf kwafeer kwayes fel zamalek?", "handwritten", "salon", "Zamalek"),
    Query("3ayzeen nefta7 ahwa w net3ashsha fe stanly", "handwritten", "cafe", "Stanley"),
    Query("fe gym ar5as fe el gam3a mansoura?", "handwritten", "gym", "University District"),
    Query("محتاجة كشف عند دكتور في مصر الجديدة", "handwritten", "clinic", "Heliopolis"),
    Query("ولادي زهقانين، فين ملاهي في القاهرة؟", "handwritten", "kids", None, "Cairo"),
    Query("فطار حلو لاتنين في الزمالك يوم الجمعة", "handwritten", "cafe", "Zamalek"),
    Query("حمام مغربي في سيدي جابر", "handwritten", "spa", "Sidi Gaber"),
    Query("كورس برمجة للمبتدئين في المنصورة", "handwritten", "course", None, "Mansoura"),
    Query(
        "بطارية الموبايل بتخلص بسرعة، تغيير بطارية في الجيزة",
        "handwritten",
        "electronics",
        None,
        "Giza",
    ),
    Query("مكان أتغدى فيه أنا وأهلي في الشيخ زايد", "handwritten", "restaurant", "Sheikh Zayed"),
    Query("فندق ويك إند في إسكندرية", "handwritten", "hotel", None, "Alexandria"),
    Query("تلميع عربية في المهندسين", "handwritten", "carwash", "Mohandessin"),
    Query("تبييض أسنان في التجمع الخامس", "handwritten", "dental", "New Cairo"),
    Query("cheap haircut maadi", "handwritten", "salon", "Maadi"),
    Query("kids birthday party giza", "handwritten", "kids", None, "Giza"),
]


# Written AFTER the failure-analysis fixes above and never used to tune anything:
# the only split that is genuinely held out. Do not edit lexicons to make these pass.
HOLDOUT: list[Query] = [
    Query("بدور على مكان أعمل فيه نضافة بشرة وسشوار في سموحة", "holdout", "salon", "Smouha"),
    Query("انا وصحابي عايزين نقعد على قهوة في الدقي", "holdout", "cafe", "Dokki"),
    Query("محتاج حد يغير شاشة الايفون في التجمع", "holdout", "electronics", "New Cairo"),
    Query("فين اقدر اعمل تحاليل دم في مدينة نصر", "holdout", "clinic", "Nasr City"),
    Query("أبغى جلسة تدليك في ستانلي", "holdout", "spa", "Stanley"),
    Query("any good buffet in mohandessin tonight", "holdout", "restaurant", "Mohandessin"),
    Query(
        "5ally el 3arabeya teb2a zay el fol, ghaseel fe zayed", "holdout", "carwash", "Sheikh Zayed"
    ),
    Query("عاوز أتعلم انجليزي، في كورسات في اسكندرية؟", "holdout", "course", None, "Alexandria"),
    Query("مكان يلعب فيه العيال في المعادي", "holdout", "kids", "Maadi"),
    Query("اشتراك نادي رياضي ٣ شهور في اكتوبر", "holdout", "gym", "6th of October"),
    Query("hotel weekend getaway mansoura", "holdout", "hotel", None, "Mansoura"),
    Query("عندي فرح الاسبوع الجاي ومحتاجة كوافير في مصر الجديدة", "holdout", "salon", "Heliopolis"),
    Query("dentist for whitening in sidi gaber", "holdout", "dental", "Sidi Gaber"),
    Query("عايز أصلح بطارية الموبايل في توريل", "holdout", "electronics", "Toriel"),
    Query("مطعم مشويات في الزمالك", "holdout", "restaurant", "Zamalek"),
]


def _metrics(ranked: list[str], rel: set[str], k: int = 10) -> dict[str, float]:
    if not rel:
        return {}
    top = ranked[:k]
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(top) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    first = next((i for i, d in enumerate(top) if d in rel), None)
    return {
        "hit@1": float(bool(top) and top[0] in rel),
        "recall@5": len(set(ranked[:5]) & rel) / min(len(rel), 5),
        "mrr@10": 0.0 if first is None else 1 / (first + 1),
        "ndcg@10": dcg / idcg,
    }


def evaluate(retriever, queries: list[Query], offers: list[Offer], mode: str) -> dict:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    t0 = time.perf_counter()
    for q in queries:
        rel = relevant(q, offers)
        if not rel:
            continue
        ranked = [h.doc_id for h in retriever.search(q.text, 10, mode)]
        by_variant[q.variant].append(_metrics(ranked, rel))
    out = {}
    for v, rows in sorted(by_variant.items()):
        out[v] = {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in rows[0]} | {
            "n": len(rows)
        }
    allrows = [r for rows in by_variant.values() for r in rows]
    out["_overall"] = {
        m: round(sum(r[m] for r in allrows) / len(allrows), 3) for m in allrows[0]
    } | {
        "n": len(allrows),
        "ms_per_query": round(1000 * (time.perf_counter() - t0) / max(1, len(allrows)), 1),
    }
    return out


# CI gate on the shipped configuration (slots). Raise these as the system improves.
GATES = {
    "templated": {"recall@5": 0.95, "mrr@10": 0.95},
    "handwritten": {"recall@5": 0.85, "mrr@10": 0.80},
    "holdout": {"recall@5": 0.70, "mrr@10": 0.70},
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true", help="skip embedding modes")
    ap.add_argument("--out", default="artifacts/eval")
    args = ap.parse_args(argv)

    offers = load()
    ids, docs = [o.offer_id for o in offers], [o.document() for o in offers]
    facets = {
        o.offer_id: {"category": o.category, "area": o.area_en, "city": o.city_en, "price": o.price}
        for o in offers
    }
    bm25 = BM25(ids, docs)
    dense = None if args.no_dense else DenseIndex(ids, docs, Embedder())
    configs = {
        "bm25_plain": (HybridRetriever(BM25(ids, docs, tokenizer=plain_tokenize)), "bm25"),
        "bm25_arabic": (HybridRetriever(bm25), "bm25"),
    }
    if dense:
        configs |= {
            "dense": (HybridRetriever(bm25, dense), "dense"),
            "hybrid_rrf": (HybridRetriever(bm25, dense), "hybrid"),
        }
    configs["slots+hybrid" if dense else "slots+bm25"] = (
        HybridRetriever(bm25, dense, facets=facets),
        "slots",
    )

    splits = {
        "templated": templated_queries(offers),
        "handwritten": HANDWRITTEN,
        "holdout": HOLDOUT,
    }
    report = {
        name: {split: evaluate(r, qs, offers, mode) for split, qs in splits.items()}
        for name, (r, mode) in configs.items()
    }

    shipped = next(n for n in report if n.startswith("slots"))
    failures = [
        f"{split} {m}={report[shipped][split]['_overall'][m]} < {floor}"
        for split, gates in GATES.items()
        for m, floor in gates.items()
        if report[shipped][split]["_overall"][m] < floor
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(
        json.dumps({"report": report, "gates": GATES, "failures": failures}, indent=2)
    )
    (out / "retrieval.md").write_text(_markdown(report, splits))
    print(_markdown(report, splits))
    print("GATE:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


def _markdown(report: dict, splits: dict) -> str:
    lines = []
    for split in splits:
        variants = sorted({v for cfg in report.values() for v in cfg[split]} - {"_overall"})
        lines.append(
            f"\n### {split} ({next(iter(report.values()))[split]['_overall']['n']} queries) - Recall@5 / MRR@10\n"
        )
        lines.append("| config | " + " | ".join(variants) + " | **overall** | ms/q |")
        lines.append("|---|" + "---|" * (len(variants) + 2))
        for name, cfg in report.items():
            cells = [
                f"{cfg[split][v]['recall@5']:.2f} / {cfg[split][v]['mrr@10']:.2f}"
                if v in cfg[split]
                else "-"
                for v in variants
            ]
            o = cfg[split]["_overall"]
            lines.append(
                f"| {name} | "
                + " | ".join(cells)
                + f" | **{o['recall@5']:.2f} / {o['mrr@10']:.2f}** | {o['ms_per_query']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
