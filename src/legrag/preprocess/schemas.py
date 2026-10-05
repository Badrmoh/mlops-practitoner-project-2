from dataclasses import dataclass, field
from enum import IntEnum
import re
from typing import Optional

class LEVEL(IntEnum):
    BOOK = 0
    CHAPTER = 1
    SECTION = 2
    TOPIC = 3
    ARTICLE = 4


@dataclass(frozen=True)
class NativeTextCell:
    text: str
    x0: float
    x1: float
    y: float
    page_no: int

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class TextLine:
    en_cells: list[NativeTextCell]
    ar_cells: list[NativeTextCell]
    # other_cells: list[NativeTextCell] | None
    barrier_x: float | None
    text_en: str
    text_ar: str
    normalized_en: str
    normalized_ar: str
    page_no: int



BOOK_RE = re.compile(r"^\s*Book\s+([\wIVXLCDM]+)\b\s*(.*)", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*Chapter\s+([\wIVXLCDM]+)\b\s*(.*)", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*Section\s+([\wIVXLCDM]+)\b\s*(.*)", re.IGNORECASE)
TOPIC_RE = re.compile(r"^\s*(\d+)\s*[.-]\s*(.*)", re.IGNORECASE)
# ARTICLE_RE = re.compile(r"^\s*Article\s+(\d+)\b\s*(.*)", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\s*Article\s+(\d+)\b(?!\.)\s*(.*)", re.IGNORECASE)
ARTICLES_REF_RE = re.compile(r"(?<!^)\bArticle\s+(\d+)\b", re.IGNORECASE)

NO_SPACE_ARTICLE_RE = re.compile(r"^\s*Article(\d{4,})\b(?!\.)\s*(.*)", re.IGNORECASE)
AR_ARTICLE_RE = re.compile(r"^\s*مادة\s+(\d+)\b\s*(.*)")
AR_REPEALED_RANGE_RE = re.compile(r"الغيت\s+المواد\s+من\s+(\d+)\s+الى\s+(\d+)")

ARTICLES_RANGE_REF_RE = re.compile(
    r"\bArticles\s+(\d+)\s*(?:to|-|–|—)\s*(\d+)\b",
    re.IGNORECASE,
)

ARTICLES_PAIR_REF_RE = re.compile(
    r"\bArticles\s+(\d+)\s+and\s+(\d+)\b",
    re.IGNORECASE,
)
LEVEL_PATTERNS = [
    (LEVEL.BOOK, BOOK_RE),
    (LEVEL.CHAPTER, CHAPTER_RE),
    (LEVEL.SECTION, SECTION_RE),
    (LEVEL.TOPIC, TOPIC_RE),
    (LEVEL.ARTICLE, ARTICLE_RE),
]

REPEALED_RANGE_RE = re.compile(
    r"\bArticles\s+(\d+)\s*[-–—]\s*(\d+)\s+(?:have\s+been\s+|were\s+)?repealed\b",
    re.IGNORECASE,
)

# ends with sentence punctuation
RE_ENDS_CLEAN = re.compile(r'[.!?]["\')\]]?\s*$')

RE_ENDS_MID = re.compile(r'[,;:—–-]\s*$')

RE_ENDS_HYPHEN = re.compile(r'\w-\s*$')

@dataclass
class RepealRange:
    start: int
    end: int

@dataclass
class Node:
    level: LEVEL
    number: str = ""
    title_en: str = ""
    title_ar: str = ""
    has_only_heading: bool = False
    children: list = field(default_factory=list)


@dataclass
class Article:
    article_number: int
    book: str
    chapter: str
    section: str
    topic: str
    text_ar: str
    text_en: str
    source_page: Optional[int]
    citation: Optional[str]
    is_repealed: bool = False
    references: list[int] = field(default_factory=list)


@dataclass
class ArticleRow:
    text_en: str
    text_ar: str
    page_no: Optional[int]
