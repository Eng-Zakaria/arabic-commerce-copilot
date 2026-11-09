"""Deterministic synthetic catalogue for "Kanz", a fictional Egyptian deals marketplace.

Everything here is invented: merchant names, areas-to-merchant pairings, prices. Same seed,
same catalogue, so retrieval metrics are reproducible across machines and CI runs.

    uv run python -m copilot.data.catalog      # writes data/offers.jsonl
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# (key, Arabic, English, Egyptian-dialect synonyms used by shoppers)
CATEGORIES: list[tuple[str, str, str, list[str]]] = [
    ("restaurant", "مطعم", "restaurant", ["مطاعم", "اكل", "أكل", "غدا", "عشا"]),
    ("cafe", "كافيه", "cafe", ["قهوة", "كافيهات", "مشروبات"]),
    ("gym", "جيم", "gym", ["صالة رياضية", "نادي", "تمرين", "اشتراك جيم"]),
    ("spa", "سبا", "spa", ["مساج", "تدليك", "جلسة سبا"]),
    ("salon", "صالون", "salon", ["كوافير", "حلاق", "شعر"]),
    ("clinic", "عيادة", "clinic", ["دكتور", "كشف", "عيادات"]),
    ("dental", "عيادة أسنان", "dental clinic", ["دكتور اسنان", "تنظيف اسنان", "تبييض"]),
    ("kids", "ملاهي أطفال", "kids play area", ["العاب اطفال", "ملاهي", "للأطفال"]),
    ("course", "كورس", "course", ["دورة", "تعليم", "كورسات", "تدريب"]),
    ("hotel", "فندق", "hotel", ["اقامة", "إقامة", "ليلة", "فنادق"]),
    ("electronics", "صيانة موبايل", "phone repair", ["تصليح موبايل", "شاشة", "صيانة"]),
    ("carwash", "غسيل عربيات", "car wash", ["غسيل سيارات", "تلميع", "عربية"]),
]

# (city_ar, city_en, [(area_ar, area_en, arabizi)])
CITIES: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "القاهرة",
        "Cairo",
        [
            ("مدينة نصر", "Nasr City", "madinet nasr"),
            ("المعادي", "Maadi", "el maadi"),
            ("الزمالك", "Zamalek", "el zamalek"),
            ("مصر الجديدة", "Heliopolis", "masr el gedida"),
            ("التجمع الخامس", "New Cairo", "el tagamo3"),
        ],
    ),
    (
        "الجيزة",
        "Giza",
        [
            ("الشيخ زايد", "Sheikh Zayed", "el sheikh zayed"),
            ("المهندسين", "Mohandessin", "el mohandseen"),
            ("الدقي", "Dokki", "el doqi"),
            ("٦ أكتوبر", "6th of October", "6 october"),
        ],
    ),
    (
        "الإسكندرية",
        "Alexandria",
        [
            ("سموحة", "Smouha", "smou7a"),
            ("ستانلي", "Stanley", "stanly"),
            ("سيدي جابر", "Sidi Gaber", "sidi gaber"),
        ],
    ),
    (
        "المنصورة",
        "Mansoura",
        [("توريل", "Toriel", "toriel"), ("الجامعة", "University District", "el gam3a")],
    ),
]

# Invented merchant name parts; combinations are fictional.
NAME_A = ["بيت", "ركن", "واحة", "نجمة", "قصر", "دار", "عالم", "سما", "لمسة", "ورد"]
NAME_A_EN = ["Beit", "Rokn", "Waha", "Negma", "Qasr", "Dar", "Alam", "Sama", "Lamsa", "Ward"]
NAME_B = [
    "النيل",
    "الياسمين",
    "الشمس",
    "القمر",
    "الهنا",
    "السعادة",
    "الأصدقاء",
    "الريف",
    "البحر",
    "الفل",
]
NAME_B_EN = [
    "El Nile",
    "El Yasmeen",
    "El Shams",
    "El Amar",
    "El Hana",
    "El Saada",
    "El Asdekaa",
    "El Reef",
    "El Bahr",
    "El Ful",
]

OFFER_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "restaurant": [
        ("وجبة عائلية لـ ٤ أفراد", "family meal for 4"),
        ("بوفيه مفتوح", "open buffet"),
        ("وجبة مشويات لفردين", "grill meal for two"),
    ],
    "cafe": [("مشروبين + حلو", "two drinks + dessert"), ("فطار لفردين", "breakfast for two")],
    "gym": [
        ("اشتراك شهر", "1-month membership"),
        ("اشتراك ٣ شهور", "3-month membership"),
        ("١٠ حصص مع مدرب", "10 sessions with a trainer"),
    ],
    "spa": [("جلسة مساج ٦٠ دقيقة", "60-minute massage"), ("حمام مغربي", "Moroccan bath")],
    "salon": [("قص وسشوار", "haircut and blow-dry"), ("صبغة شعر", "hair colouring")],
    "clinic": [("كشف + استشارة", "check-up + consultation"), ("تحاليل شاملة", "full blood panel")],
    "dental": [
        ("تنظيف وتلميع أسنان", "teeth cleaning and polishing"),
        ("تبييض أسنان", "teeth whitening"),
    ],
    "kids": [("تذكرة يوم كامل", "full-day ticket"), ("حفلة عيد ميلاد", "birthday party package")],
    "course": [
        ("كورس إنجليزي شهر", "1-month English course"),
        ("كورس برمجة للمبتدئين", "beginner coding course"),
    ],
    "hotel": [
        ("ليلة لشخصين شامل الفطار", "one night for two with breakfast"),
        ("ويك إند ليلتين", "two-night weekend"),
    ],
    "electronics": [
        ("تغيير شاشة موبايل", "phone screen replacement"),
        ("تغيير بطارية", "battery replacement"),
    ],
    "carwash": [
        ("غسيل داخلي وخارجي", "interior and exterior wash"),
        ("تلميع وحماية", "polish and protection"),
    ],
}

PRICE_RANGE = {
    "restaurant": (200, 1200),
    "cafe": (80, 400),
    "gym": (300, 3000),
    "spa": (300, 1500),
    "salon": (100, 900),
    "clinic": (200, 1500),
    "dental": (300, 3000),
    "kids": (100, 2500),
    "course": (500, 4000),
    "hotel": (1500, 9000),
    "electronics": (300, 4000),
    "carwash": (100, 700),
}


@dataclass
class Offer:
    offer_id: str
    merchant_ar: str
    merchant_en: str
    category: str
    category_ar: str
    city_ar: str
    city_en: str
    area_ar: str
    area_en: str
    area_arabizi: str
    title_ar: str
    title_en: str
    original_price: int
    discount_pct: int
    price: int
    valid_until: str
    tags: list[str] = field(default_factory=list)

    def text_ar(self) -> str:
        return f"{self.title_ar} من {self.merchant_ar} - {self.category_ar} في {self.area_ar}، {self.city_ar}"

    def text_en(self) -> str:
        return f"{self.title_en} at {self.merchant_en} - {self.category} in {self.area_en}, {self.city_en}"

    def document(self) -> str:
        """What the retriever indexes: both languages, plus how shoppers spell the area."""
        return " | ".join([self.text_ar(), self.text_en(), self.area_arabizi, " ".join(self.tags)])


def build_catalog(n: int = 400, seed: int = 7, today: date = date(2026, 11, 1)) -> list[Offer]:
    rng = random.Random(seed)
    offers: list[Offer] = []
    seen_merchants: set[tuple[str, str]] = set()
    i = 0
    while len(offers) < n:
        cat, cat_ar, _cat_en, synonyms = CATEGORIES[i % len(CATEGORIES)]
        city_ar, city_en, areas = rng.choice(CITIES)
        area_ar, area_en, area_arabizi = rng.choice(areas)
        a, b = rng.randrange(len(NAME_A)), rng.randrange(len(NAME_B))
        merchant_ar = f"{NAME_A[a]} {NAME_B[b]}"
        merchant_en = f"{NAME_A_EN[a]} {NAME_B_EN[b]}"
        key = (merchant_en, area_en)
        i += 1
        if key in seen_merchants and rng.random() < 0.7:
            continue  # keep merchant/area pairs mostly unique, like a real marketplace
        seen_merchants.add(key)
        title_ar, title_en = rng.choice(OFFER_TEMPLATES[cat])
        lo, hi = PRICE_RANGE[cat]
        original = rng.randrange(lo, hi, 10)
        discount = rng.choice([15, 20, 25, 30, 35, 40, 50])
        offers.append(
            Offer(
                offer_id=f"KZ-{len(offers) + 1:04d}",
                merchant_ar=merchant_ar,
                merchant_en=merchant_en,
                category=cat,
                category_ar=cat_ar,
                city_ar=city_ar,
                city_en=city_en,
                area_ar=area_ar,
                area_en=area_en,
                area_arabizi=area_arabizi,
                title_ar=title_ar,
                title_en=title_en,
                original_price=original,
                discount_pct=discount,
                price=round(original * (100 - discount) / 100 / 5) * 5,
                valid_until=(today + timedelta(days=rng.randrange(7, 120))).isoformat(),
                tags=rng.sample(synonyms, k=min(2, len(synonyms))),
            )
        )
    return offers


def save(offers: list[Offer], path: Path | None = None) -> Path:
    path = path or DATA_DIR / "offers.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in offers:
            f.write(json.dumps(asdict(o), ensure_ascii=False) + "\n")
    return path


def load(path: Path | None = None) -> list[Offer]:
    path = path or DATA_DIR / "offers.jsonl"
    if not path.exists():
        save(build_catalog(), path)
    with path.open(encoding="utf-8") as f:
        return [Offer(**json.loads(line)) for line in f if line.strip()]


if __name__ == "__main__":
    p = save(build_catalog())
    print(f"wrote {p}")
