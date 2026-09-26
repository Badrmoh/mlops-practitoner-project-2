import re


# Arabic short vowels and annotation marks that should not affect matching.
TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

# Parentheses appear around article markers in Arabic text: "( مادة ١ (".
PARENTHESES_RE = re.compile(r"[()（）]")

# Normalize Alef forms that carry Hamza/Madda/Wasla to bare Alef.
ALEF_TRANSLATION = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
})

# Re-collapse spaces after deleting punctuation/diacritics.
WHITESPACE_RE = re.compile(r"\s+")

DIGIT_TRANSLATION = str.maketrans({
    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
})

DIGIT_RUN_RE = re.compile(r"\d+")

def normalize_digits(text: str) -> str:
    text = text.translate(DIGIT_TRANSLATION)
    text = DIGIT_RUN_RE.sub(lambda match: match.group(0)[::-1], text)
    return text.translate(DIGIT_TRANSLATION)

def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def remove_parentheses(text: str) -> str:
    return PARENTHESES_RE.sub(" ", text)


def remove_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text)


def normalize_alef(text: str) -> str:
    return text.translate(ALEF_TRANSLATION)


def normalize_arabic(text: str) -> str:
    text = remove_parentheses(text)
    text = remove_tashkeel(text)
    text = normalize_alef(text)
    text = normalize_whitespace(text)
    return normalize_digits(text)


def normalize_english(text: str) -> str:
    text = remove_parentheses(text)
    return normalize_whitespace(text)
