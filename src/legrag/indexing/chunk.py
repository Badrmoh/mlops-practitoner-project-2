import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from legrag.config import Settings

NUMBERED_CHUNK_RE = re.compile(r"(?<=[.!؟:؛])\s+(?=\d+\b)")
UNNUMBERED_CHUNK_RE = re.compile(r"(?<=[.!؟])\s+")


@dataclass
class Chunk:
    text: str
    metadata: dict


class Chunker:
    def __init__(self, config: Settings) -> None:
        self.config: Settings = config if config else Settings()
        self._max_chunk_chars_total: int = self.appr_tokens_to_chars(self.config.max_tokens_per_chunk)
        self._chunk_overlap_chars: int = self.appr_tokens_to_chars(self.config.chunk_overlap_tokens)
        self._max_chunk_chars: int = self._max_chunk_chars_total - self._chunk_overlap_chars

    def load_articles_json(self) -> list[dict]:
        with open(self.config.articles_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def appr_tokens_to_chars(self, tokens: int) -> int:
        return int(tokens * self.config.approx_chars_per_token)

    def get_overlap_head(self, words: list[str], max_chars: int) -> list[str]:
        selected: list[str] = []
        total = 0

        for word in words:
            extra = len(word) + (1 if selected else 0)
            if total + extra > max_chars:
                break
            selected.append(word)
            total += extra

        return selected

    def get_overlap_tail(self, words: list[str], max_chars: int) -> list[str]:
        selected: list[str] = []
        total = 0

        for word in reversed(words):
            extra = len(word) + (1 if selected else 0)
            if total + extra > max_chars:
                break
            selected.append(word)
            total += extra

        return list(reversed(selected))

    def chunk_text_at_word_boundary(self, text: str) -> list[str]:
        words: list[str] = text.split()
        if not words:
            return []

        chunks_count: int = max(1, math.ceil(len(text) / self._max_chunk_chars))
        chunk_word_count: int = max(1, math.ceil(len(words) / chunks_count))

        word_chunks: list[list[str]] = [
            words[i:i + chunk_word_count]
            for i in range(0, len(words), chunk_word_count)
        ]

        chunks: list[str] = []

        for index, current_chunk_words in enumerate(word_chunks):
            previous_chunk_words = word_chunks[index - 1] if index > 0 else []
            next_chunk_words = word_chunks[index + 1] if index + 1 < len(word_chunks) else []

            overlap_tail = self.get_overlap_tail(previous_chunk_words, self._chunk_overlap_chars)
            overlap_head = self.get_overlap_head(next_chunk_words, self._chunk_overlap_chars)

            chunk_words = overlap_tail + current_chunk_words + overlap_head
            chunks.append(" ".join(chunk_words))

        return chunks

    def chunk_article_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        if len(text) > self._max_chunk_chars_total:
            chunks: list[str] = NUMBERED_CHUNK_RE.split(text)
            if len(chunks) == 1:
                chunks = UNNUMBERED_CHUNK_RE.split(text)
            if len(chunks) == 1:
                chunks = self.chunk_text_at_word_boundary(text)
        else:
            chunks.append(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]


    def create_chunks(self, articles: list[dict]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for article in articles:
            chunks_en: list[str] = []
            chunks_ar: list[str] = []
            metadata={
                "article_number": article["article_number"],
                "book": article["book"],
                "chapter": article["chapter"],
                "section": article["section"],
                "topic": article["topic"],
                "source_page": article["source_page"],
                "citation": article["citation"],
                "is_repealed": article["is_repealed"],
                "referenced_articles": article["references"] or "none",
            }
            if len(article["text_en"]) > self._max_chunk_chars:
                chunks_en = self.chunk_article_text(article["text_en"])
                for i, chunk in enumerate(chunks_en):
                    eng_doc = Chunk(
                        text=chunk,
                        metadata={
                            **metadata,
                            "chunk_index": i,
                            "language": "en",
                            "chunk_count": len(chunks_en),
                            "chunk_id": (
                                f"article-"
                                f"{article["article_number"]}-"
                                f"en-{i}"
                            )
                        }
                    )
                    chunks.append(eng_doc)
            else:
                eng_doc = Chunk(
                    text=article["text_en"],
                    metadata={
                        **metadata,
                        "language": "en",
                        "chunk_index": 0,
                        "chunk_count": 1,
                        "chunk_id": (
                            f"article-"
                            f"{article["article_number"]}-"
                            f"en-0"
                        ),
                    }
                )
                chunks.append(eng_doc)

            if len(article["text_ar"]) > self._max_chunk_chars:
                chunks_ar = self.chunk_article_text(article["text_ar"])
                for i, chunk in enumerate(chunks_ar):
                    ar_doc = Chunk(
                        text=chunk,
                        metadata={
                            **metadata,
                            "chunk_index": i,
                            "language": "ar",
                            "chunk_count": len(chunks_ar),
                            "chunk_id": (
                                f"article-"
                                f"{article["article_number"]}-"
                                f"ar-{i}"
                            )
                        }
                    )
                    chunks.append(ar_doc)
            else:
                ar_doc = Chunk(
                    text=article["text_ar"],
                    metadata={
                        **metadata,
                        "language": "ar",
                        "chunk_index": 0,
                        "chunk_count": 1,
                        "chunk_id": (
                            f"article-"
                            f"{article["article_number"]}-"
                            f"ar-0"
                        ),
                    }
                )
                chunks.append(ar_doc)

        return chunks

    def run(self) -> list[Chunk]:
        articles = self.load_articles_json()
        return self.create_chunks(articles)
