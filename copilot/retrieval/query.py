"""Query understanding: pull structured slots out of a free-text shopping query.

"عايز جيم رخيص في مدينة نصر تحت ٥٠٠"  ->  category=gym, area=Nasr City, max_price=500, sort=price
"3ayez mat3am fe smou7a"              ->  category=restaurant, area=Smouha
"spa in Alex"                         ->  category=spa, city=Alexandria

Lexicons are built from the catalogue's own vocabulary (Arabic, English, Arabizi, dialect
synonyms), all compared in normalised form, so a new area or category needs no code change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data.catalog import CATEGORIES, CITIES
from ..text.arabic import arabizi_to_arabic, looks_like_arabizi, normalize

# Extra English/Arabizi words shoppers use for each category.
CATEGORY_EXTRA: dict[str, list[str]] = {
    "restaurant": [
        "food",
        "lunch",
        "dinner",
        "meal",
        "grill",
        "buffet",
        "mat3am",
        "akl",
        "ma6a3em",
        "اتغدي",
        "نتغدي",
        "اتعشي",
        "نتعشي",
        "هنتعشي",
        "هنتغدي",
        "مشويات",
    ],
    "cafe": [
        "coffee",
        "breakfast",
        "ahwa",
        "kafeh",
        "coffee shop",
        "فطار",
        "فطور",
        "افطر",
        "fetar",
    ],
    "gym": ["fitness", "workout", "membership", "gim", "trainer"],
    "spa": ["massage", "masa", "moroccan bath", "حمام مغربي"],
    "salon": ["haircut", "hair", "barber", "kwafeer", "7ala2"],
    "clinic": ["doctor", "checkup", "check-up", "dktor", "تحاليل", "lab test"],
    "dental": [
        "dentist",
        "teeth",
        "whitening",
        "asnan",
        "sinan",
        "اسنان",
        "أسنان",
        "سنان",
        "سناني",
        "دكتور سنان",
        "دكتور اسنان",
        "تنضيف سنان",
    ],
    "kids": ["kids", "children", "birthday", "atfal", "malahy", "اطفال", "أطفال"],
    "course": ["english course", "coding", "programming", "kors", "class", "انجليزي", "برمجة"],
    "hotel": ["night", "stay", "weekend", "fondo2", "fondoq", "room"],
    "electronics": ["screen", "battery", "repair", "mobile", "shasha", "موبايل", "شاشة", "بطارية"],
    "carwash": ["car", "wash", "detailing", "3arabeya", "ghasil", "عربية", "سيارة"],
}
CITY_EXTRA: dict[str, list[str]] = {
    "Cairo": ["masr", "cairo", "مصر"],
    "Giza": ["giza", "el giza"],
    "Alexandria": ["alex", "eskendereya", "اسكندرية", "إسكندرية"],
    "Mansoura": ["mansoura", "el mansoura"],
}
# Everyday short names people use instead of the full area name.
AREA_EXTRA: dict[str, list[str]] = {
    "New Cairo": ["التجمع", "tagamo3", "new cairo", "fifth settlement"],
    "6th of October": ["اكتوبر", "october", "6 october city"],
    "Sheikh Zayed": ["زايد", "zayed"],
    "Heliopolis": ["heliopolis", "masr el gedida"],
    "Nasr City": ["nasr city", "مدينه نصر"],
    "University District": ["gam3a", "الجامعه"],
}
CHEAP_WORDS = {"رخيص", "ارخص", "اوفر", "اقل سعر", "cheap", "cheapest", "budget", "ar5as", "r5ees"}
_PRICE = re.compile(r"(?:تحت|اقل من|لحد|under|below|less than|max|ta7t)\s*(\d{2,6})")


def _variants(*words: str) -> set[str]:
    out = set()
    for w in words:
        n = normalize(w)
        out.add(n)
        if n.startswith("ال") and len(n) > 4:
            out.add(n[2:])
        if n.startswith("el ") and len(n) > 5:
            out.add(n[3:])
    return out


def _build_lexicons():
    cats: dict[str, set[str]] = {}
    for key, ar, en, syns in CATEGORIES:
        cats[key] = _variants(ar, en, *syns, *CATEGORY_EXTRA.get(key, []))
    areas: dict[str, tuple[str, set[str]]] = {}
    cities: dict[str, set[str]] = {}
    for city_ar, city_en, area_list in CITIES:
        cities[city_en] = _variants(city_ar, city_en, *CITY_EXTRA.get(city_en, []))
        for area_ar, area_en, arabizi in area_list:
            areas[area_en] = (
                city_en,
                _variants(area_ar, area_en, arabizi, *AREA_EXTRA.get(area_en, [])),
            )
    return cats, areas, cities


CATEGORY_LEXICON, AREA_LEXICON, CITY_LEXICON = _build_lexicons()


@dataclass
class Slots:
    category: str | None = None
    area: str | None = None
    city: str | None = None
    max_price: int | None = None
    cheapest: bool = False
    matched: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.category or self.area or self.city or self.max_price)


def _surface(query: str) -> str:
    """Normalised query plus Arabic renderings of any Arabizi words, padded for whole-word matching."""
    norm = re.sub(r"[^\w\s]", " ", normalize(query))  # "zamalek?" must still match "zamalek"
    extra = [
        normalize(arabizi_to_arabic(t))
        for t in norm.split()
        if t.isascii() and looks_like_arabizi(t)
    ]
    return f" {norm} {' '.join(extra)} "


def _find(surface: str, terms: set[str]) -> str | None:
    best = None
    for term in terms:
        for t in (term, "ال" + term if not term.isascii() else term):
            if f" {t} " in surface and (best is None or len(t) > len(best)):
                best = t
    return best


def parse(query: str) -> Slots:
    s = Slots()
    surface = _surface(query)
    # longest match wins, so "عيادة أسنان" beats "عيادة" and "مدينة نصر" beats "نصر"
    best_cat = max(
        ((k, m) for k, terms in CATEGORY_LEXICON.items() if (m := _find(surface, terms))),
        key=lambda km: len(km[1]),
        default=None,
    )
    if best_cat:
        s.category, _ = best_cat
        s.matched.append(f"category:{best_cat[1]}")
    best_area = max(
        ((a, c, m) for a, (c, terms) in AREA_LEXICON.items() if (m := _find(surface, terms))),
        key=lambda acm: len(acm[2]),
        default=None,
    )
    if best_area:
        s.area, s.city, _ = best_area
        s.matched.append(f"area:{best_area[2]}")
    elif city := next((c for c, terms in CITY_LEXICON.items() if _find(surface, terms)), None):
        s.city = city
        s.matched.append(f"city:{city}")
    if m := _PRICE.search(surface):
        s.max_price = int(m.group(1))
    s.cheapest = any(f" {normalize(w)} " in surface for w in CHEAP_WORDS)
    return s
