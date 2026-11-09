"""Arabic text handling for search: orthographic normalisation, Egyptian dialect, Arabizi.

Shoppers type the same word many ways: أحسن / احسن, مدرسة / مدرسه, على / علي, with or
without diacritics, in Arabizi ("3ayez gym fe madinet nasr"), or with Egyptian vocabulary
("عايز", "فين", "أرخص"). Retrieval only works if all of those collapse to one form.

Every function is pure and deterministic so it can be unit-tested and reused identically
at index time and query time.
"""

from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------- orthography ---
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",  # alef variants
        "ى": "ي",  # alef maqsura -> ya (Egyptian writing mixes them)
        "ة": "ه",  # ta marbuta -> ha
        "ؤ": "و",
        "ئ": "ي",  # hamza carriers
        "گ": "ك",
        "ک": "ك",
        "ی": "ي",  # Persian/Urdu lookalikes from some keyboards
    }
)
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


def normalize(text: str) -> str:
    """Canonical form for matching: no diacritics/tatweel, unified letters, ASCII digits."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_LETTER_MAP).translate(_EASTERN_DIGITS)
    # "حلووووو" -> "حلوو". Letters only: digits must survive ("10001", "1000" are ids and prices).
    text = re.sub(r"([^\W\d_])\1{2,}", r"\1\1", text)
    return text.lower()


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_CHAR.search(text))


# -------------------------------------------------------------- Egyptian dialect ---
# Dialect -> words the catalogue actually uses. Expansion adds terms; it never removes the
# original, so precision on standard queries is unaffected.
DIALECT_SYNONYMS: dict[str, list[str]] = {
    "عايز": ["اريد"],
    "عاوز": ["اريد"],
    "عايزه": ["اريد"],
    "محتاج": ["اريد"],
    "فين": ["اين"],
    "ازاي": ["كيف"],
    "امتي": ["متي"],
    "ليه": ["لماذا"],
    "ارخص": ["خصم", "سعر"],
    "رخيص": ["خصم", "سعر"],
    "اوفر": ["خصم", "عرض"],
    "عربيه": ["سياره"],
    "عربيات": ["سيارات"],
    "موبايل": ["هاتف"],
    "تليفون": ["موبايل", "هاتف"],
    "كوافير": ["صالون"],
    "حلاق": ["صالون"],
    "كشف": ["عياده"],
    "دكتور": ["عياده"],
    "اكل": ["مطعم", "وجبه"],
    "غدا": ["مطعم", "وجبه"],
    "عشا": ["مطعم", "وجبه"],
    "قهوه": ["كافيه"],
    "نادي": ["جيم"],
    "تمرين": ["جيم"],
    "مساج": ["سبا"],
    "تدليك": ["سبا"],
    "ملاهي": ["اطفال"],
    "عيال": ["اطفال"],
    "ولاد": ["اطفال"],
    "لعب": ["العاب"],
    "اوضه": ["غرفه", "فندق"],
    "اقامه": ["فندق"],
    "تصليح": ["صيانه"],
    "شاشه": ["صيانه", "موبايل"],
    "كورس": ["دوره"],
    "دوره": ["كورس"],
    "انجلش": ["انجليزي"],
    "بروجرامنج": ["برمجه"],
    "اسكندريه": ["الاسكندريه"],
    "التجمع": ["التجمع الخامس"],
    "اكتوبر": ["6 اكتوبر"],
}

# Stopwords that carry no product meaning in shopping queries (normalised forms).
STOPWORDS = {
    "في",
    "من",
    "علي",
    "الي",
    "عن",
    "مع",
    "او",
    "و",
    "ده",
    "دي",
    "انا",
    "عايز",
    "عاوز",
    "اريد",
    "ممكن",
    "لو",
    "سمحت",
    "عرض",
    "عروض",
    "فيه",
    "بتاع",
    "بتاعت",
    "اي",
    "ايه",
    "حاجه",
    "the",
    "a",
    "an",
    "in",
    "at",
    "for",
    "of",
    "to",
    "i",
    "want",
    "need",
    "any",
    "some",
    "please",
    "offer",
    "offers",
    "deal",
    "deals",
    "me",
    "fe",
    "fy",
    "3ayez",
    "3awez",
    "3ard",
    "3orood",
    "3rood",
}


def _strip_al(token: str) -> str:
    """Light stemming: the definite article and common proclitics (و، ب، ل، ف + ال)."""
    for prefix in ("وال", "بال", "لل", "فال", "كال", "ال"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            return token[len(prefix) :]
    return token


# --------------------------------------------------------------------- Arabizi ---
# Word-level dictionary for frequent shopping words, then a character fallback.
ARABIZI_WORDS: dict[str, str] = {
    "3ayez": "عايز",
    "3awez": "عاوز",
    "3ayza": "عايزه",
    "ana": "انا",
    "fe": "في",
    "fy": "في",
    "feen": "فين",
    "fen": "فين",
    "ezay": "ازاي",
    "ar5as": "ارخص",
    "ar5s": "ارخص",
    "r5ees": "رخيص",
    "3ard": "عرض",
    "3orood": "عروض",
    "mat3am": "مطعم",
    "ma6a3em": "مطاعم",
    "akl": "اكل",
    "kafeh": "كافيه",
    "ahwa": "قهوه",
    "gym": "جيم",
    "gim": "جيم",
    "masa": "مساج",
    "massage": "مساج",
    "spa": "سبا",
    "kwafeer": "كوافير",
    "7ala2": "حلاق",
    "doctor": "دكتور",
    "dktor": "دكتور",
    "asnan": "اسنان",
    "sinan": "اسنان",
    "atfal": "اطفال",
    "malahy": "ملاهي",
    "kors": "كورس",
    "course": "كورس",
    "fondo2": "فندق",
    "fondoq": "فندق",
    "mobile": "موبايل",
    "mobayl": "موبايل",
    "shasha": "شاشه",
    "ta9le7": "تصليح",
    "tasle7": "تصليح",
    "3arabeya": "عربيه",
    "ghasil": "غسيل",
    "8aseel": "غسيل",
    "ghaseel": "غسيل",
    "madinet": "مدينه",
    "nasr": "نصر",
    "maadi": "المعادي",
    "zamalek": "الزمالك",
    "tagamo3": "التجمع",
    "zayed": "زايد",
    "sheikh": "الشيخ",
    "mohandseen": "المهندسين",
    "doqi": "الدقي",
    "dokki": "الدقي",
    "october": "اكتوبر",
    "smou7a": "سموحه",
    "smouha": "سموحه",
    "stanly": "ستانلي",
    "gaber": "جابر",
    "sidi": "سيدي",
    "toriel": "توريل",
    "gam3a": "الجامعه",
    "eskendereya": "اسكندريه",
    "masr": "مصر",
    "gedida": "الجديده",
    "el": "ال",
    "w": "و",
    "b": "ب",
    "bel": "بال",
}

_ARABIZI_DIGIT = re.compile(r"[a-z]*[2356789][a-z0-9]*")
_ARABIZI_CHARS = [
    ("sh", "ش"),
    ("kh", "خ"),
    ("gh", "غ"),
    ("th", "ث"),
    ("dh", "ذ"),
    ("2", "ء"),
    ("3", "ع"),
    ("5", "خ"),
    ("6", "ط"),
    ("7", "ح"),
    ("8", "غ"),
    ("9", "ص"),
    ("a", "ا"),
    ("b", "ب"),
    ("t", "ت"),
    ("g", "ج"),
    ("d", "د"),
    ("r", "ر"),
    ("z", "ز"),
    ("s", "س"),
    ("f", "ف"),
    ("q", "ق"),
    ("k", "ك"),
    ("l", "ل"),
    ("m", "م"),
    ("n", "ن"),
    ("h", "ه"),
    ("w", "و"),
    ("y", "ي"),
    ("e", "ي"),
    ("i", "ي"),
    ("o", "و"),
    ("u", "و"),
]


def looks_like_arabizi(token: str) -> bool:
    """Latin word with Arabizi digits (3, 7, 2, 5 ...) or a known Arabizi word."""
    return token in ARABIZI_WORDS or bool(_ARABIZI_DIGIT.fullmatch(token))


def arabizi_to_arabic(token: str) -> str:
    if token in ARABIZI_WORDS:
        return ARABIZI_WORDS[token]
    out, i = [], 0
    while i < len(token):
        for latin, arabic in _ARABIZI_CHARS:
            if token.startswith(latin, i):
                out.append(arabic)
                i += len(latin)
                break
        else:
            i += 1
    return "".join(out)


# -------------------------------------------------------------------- tokenize ---
def tokenize(text: str, *, expand: bool = True) -> list[str]:
    """Search tokens: normalised, article-stripped, Arabizi transliterated, dialect-expanded.

    Latin words are kept as-is (English matches the English title) and, when they look
    like Arabizi, also added in Arabic script.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(normalize(text)):
        if raw.isascii() and looks_like_arabizi(raw):
            tokens.append(raw)
            ar = normalize(arabizi_to_arabic(raw))
            if ar:
                tokens.extend(_strip_al(t) for t in ar.split())
            continue
        tok = _strip_al(raw)
        tokens.append(tok)
        if expand and raw in DIALECT_SYNONYMS:
            for syn in DIALECT_SYNONYMS[raw]:
                tokens.extend(_strip_al(t) for t in normalize(syn).split())
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]


def plain_tokenize(text: str) -> list[str]:
    """Baseline used in the eval ablation: lowercase + whitespace/punctuation split only."""
    return [t for t in _TOKEN.findall(text.lower()) if t]


def reply_language(text: str) -> str:
    """'ar' for Arabic script or Arabizi (a shopper typing Arabizi expects an Egyptian Arabic
    reply), else 'en'."""
    if has_arabic(text):
        return "ar"
    words = _TOKEN.findall(text.lower())
    arabizi = sum(
        1
        for w in words
        if looks_like_arabizi(w)
        and w not in {"gym", "spa", "course", "mobile", "massage", "doctor"}
    )
    return "ar" if words and arabizi / len(words) >= 0.3 else "en"
