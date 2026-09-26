import json
import logging
import re
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem

from legrag.ingestion.native_rows import extract_native_cells, group_cells_by_line
from legrag.ingestion.normalize import normalize_english
from legrag.ingestion.schemas import (
    ARTICLE_RE,
    ARTICLES_PAIR_REF_RE,
    ARTICLES_RANGE_REF_RE,
    ARTICLES_REF_RE,
    LEVEL,
    LEVEL_PATTERNS,
    RE_ENDS_CLEAN,
    REPEALED_RANGE_RE,
    NO_SPACE_ARTICLE_RE,
    AR_ARTICLE_RE,
    AR_REPEALED_RANGE_RE,
    Article,
    ArticleRow,
    Node,
    TextLine,
)


def text_lines_to_article_rows(lines: list[TextLine]) -> list[ArticleRow]:
    return [
        ArticleRow(
            text_en=line.text_en,
            text_ar=line.text_ar,
            page_no=line.page_no,
        )
        for line in lines
        if line.text_en or line.text_ar
    ]


def iter_native_rows(doc: DoclingDocument) -> list[ArticleRow]:
    cells = extract_native_cells(doc)
    lines = group_cells_by_line(cells)
    return text_lines_to_article_rows(lines)


def current_article_ends_clean(rows: list[ArticleRow]) -> bool:
    text_en = " ".join(row.text_en.strip() for row in rows if row.text_en.strip())
    return bool(RE_ENDS_CLEAN.search(text_en))


def split_article_start(
    text_en: str,
    text_ar: str = "",
    expected_next_article_number: int | None = None,
    current_article_number: int | None = None,
) -> tuple[int, str, str] | None:
    en_match = ARTICLE_RE.match(text_en) or NO_SPACE_ARTICLE_RE.match(text_en)
    ar_match = AR_ARTICLE_RE.match(text_ar)

    # This step is a quality gate just to make sure
    # that extraction of articles is accurate
    if en_match and ar_match:
        en_number = int(en_match.group(1))
        ar_number = int(ar_match.group(1))
        if en_number == ar_number:
            return en_number, en_match.group(2).strip(), ar_match.group(2).strip()
        return en_number, en_match.group(2).strip(), text_ar.strip()

    # Matching english is always more reliable so we
    # depend on it primarily
    if en_match:
        return int(en_match.group(1)), en_match.group(2).strip(), text_ar.strip()

    if ar_match:
        ar_number = int(ar_match.group(1))
        is_expected = (
            ar_number == expected_next_article_number or
            (
                current_article_number is not None and
                ar_number == current_article_number + 1
            )
        )
        if is_expected:
            return ar_number, text_en.strip(), ar_match.group(2).strip()

    # Cell does not start a new article.
    return None


def detect_repealed_range(text: str) -> tuple[int, int] | None:
    if match := REPEALED_RANGE_RE.search(text):
        return int(match.group(1)), int(match.group(2))
    if match := AR_REPEALED_RANGE_RE.search(text):
        return int(match.group(1)), int(match.group(2))
    return None


def looks_like_title(text: str) -> bool:
    # Starts with a capital, and has at least 33% of words starting with a capital
    words = text.split()
    capitalized = sum(1 for word in words if word[:1].isupper())
    return capitalized >= max(1, len(words) // 3) and not text.endswith(".")


def classify(
    text_en: str,
    text_ar: str = "",
) -> Node | None:
    """Classify English row text into document hierarchy level."""
    # Normalize leading/trailing whitespace before matching hierarchy patterns.
    text_en = text_en.strip()

    # Try each known level pattern in order: book, chapter, section, topic, article.
    for level, pattern in LEVEL_PATTERNS:
        # If current row matches this level, convert it into a hierarchy node.
        if match := pattern.match(text_en):
            # Most patterns capture number in group 1 and remaining title in group 2.
            title_en = match.group(2).strip() if len(match.groups()) > 1 else ""
            # Node stores detected hierarchy level plus English/Arabic labels.
            return Node(
                level=level,
                number=match.group(1),
                title_en=title_en,
                title_ar=text_ar.strip(),
            )

    # Row is not a recognized heading/article marker.
    return None


def iter_table_rows(doc: DoclingDocument) -> list[ArticleRow]:
    """Create left-to-right bilingual table rows from Docling tables."""
    # Final list of normalized bilingual rows from all tables.
    rows: list[ArticleRow] = []

    # Docling document contains many item types; ingestion only needs tables.
    for item, _ in doc.iterate_items():
        # Skip non-table items.
        if not isinstance(item, TableItem):
            continue

        # Page number comes from table provenance when Docling provides it.
        page_no = item.prov[0].page_no if item.prov else None

        # Group Docling table cells by row index.
        table_rows: dict[int, list] = {}
        for cell in item.data.table_cells:
            # Cells with same start_row_offset_idx belong to same table row.
            table_rows.setdefault(cell.start_row_offset_idx, []).append(cell)

        # Convert each grouped table row into one ArticleRow.
        for row_cells in table_rows.values():
            # Sort by column offset so left cell comes first and right cells follow.
            ordered = sorted(row_cells, key=lambda cell: cell.start_col_offset_idx)

            # In this PDF, leftmost cell is English.
            text_en = ordered[0].text.strip() if ordered else ""

            # Remaining cells are Arabic/right side; join them if Docling split columns.
            text_ar = " ".join(cell.text.strip() for cell in ordered[1:] if cell.text.strip())

            # Keep row only if either language has content.
            if text_en or text_ar:
                rows.append(ArticleRow(text_en=text_en, text_ar=text_ar, page_no=page_no))

    # Return document rows in the order Docling yielded tables/rows.
    return rows


def update_hierarchy(hierarchy: dict[LEVEL, Node], node: Node) -> None:
    """Keep latest node at current level and discard stale descendants."""
    # Store latest heading for its level.
    hierarchy[node.level] = node

    # If parent level changes, deeper child levels from old parent are no longer valid.
    for level in list(hierarchy):
        # LEVEL enum order is parent-to-child: BOOK < CHAPTER < SECTION < TOPIC < ARTICLE.
        if level > node.level:
            # Remove stale child heading.
            del hierarchy[level]


def extract_article_references(text: str) -> list[int]:
    text = text.strip()
    refs: list[int] = []

    for start, end in ARTICLES_RANGE_REF_RE.findall(text):
        refs.extend(range(int(start), int(end) + 1))

    for first, second in ARTICLES_PAIR_REF_RE.findall(text):
        refs.extend([int(first), int(second)])

    refs.extend(int(ref) for ref in ARTICLES_REF_RE.findall(text))

    return list(dict.fromkeys(refs))


def hierarchy_title(hierarchy: dict[LEVEL, Node], level: LEVEL) -> str:
    node = hierarchy.get(level)
    if not node:
        return ""
    return node.title_en.strip()


def build_repealed_article(
    number: int,
    source_page: int | None,
    hierarchy: dict[LEVEL, Node],
) -> Article:
    return Article(
        article_number=number,
        book=hierarchy_title(hierarchy, LEVEL.BOOK),
        chapter=hierarchy_title(hierarchy, LEVEL.CHAPTER),
        section=hierarchy_title(hierarchy, LEVEL.SECTION),
        topic=hierarchy_title(hierarchy, LEVEL.TOPIC),
        text_ar=f"مادة {number} ملغاة",
        text_en=f"Article {number} no longer exists. It was repealed.",
        source_page=source_page,
        citation=f"Egyptian Civil Code, Article {number}",
        is_repealed=True,
    )


def build_article(number: int, rows: list[ArticleRow], hierarchy: dict[LEVEL, Node]) -> Article:
    # Join all English continuation rows belonging to this article.
    text_en = normalize_english(" ".join(row.text_en.strip() for row in rows if row.text_en.strip()))

    # Join all Arabic continuation rows belonging to this article.
    text_ar = " ".join(row.text_ar.strip() for row in rows if row.text_ar.strip())

    # Build final article record used by downstream JSON/export code.
    return Article(
        article_number=number,
        book=hierarchy_title(hierarchy, LEVEL.BOOK),
        chapter=hierarchy_title(hierarchy, LEVEL.CHAPTER),
        section=hierarchy_title(hierarchy, LEVEL.SECTION),
        topic=hierarchy_title(hierarchy, LEVEL.TOPIC),
        text_ar=text_ar,
        text_en=text_en,
        source_page=rows[0].page_no if rows else None,
        citation=f"Egyptian Civil Code, Article {number}",
        is_repealed=False,
        references=extract_article_references(text_en.strip())
    )


def create_repealed_articles(
    articles: list[Article],
    start: int,
    end: int,
    source_page: int | None,
    hierarchy: dict[LEVEL, Node],
):
    existing_numbers = {article.article_number for article in articles}
    for number in range(start, end + 1):
        if number not in existing_numbers:
            articles.append(build_repealed_article(number, source_page, hierarchy))


def close_article(
    articles: list[Article],
    current_article_number: int,
    current_article_lines: list[ArticleRow],
    current_hierarchy: dict[LEVEL, Node],
) -> int | None:
    """
    Close open article
    1. Group article rows and build into Article object]
    2. Detect if the this article states repealed articles range
    3. Return the detected repealed range if detected
    """

    article = build_article(current_article_number, current_article_lines, current_hierarchy)
    articles.append(article)

    repealed_range = detect_repealed_range(f"{article.text_en} {article.text_ar}")
    if not repealed_range:
        return None

    start, end = repealed_range
    create_repealed_articles(articles, start, end, article.source_page, current_hierarchy)

    return end + 1


def close_current_article_if_clean(
    articles: list[Article],
    current_article_number: int | None,
    current_article_lines: list[ArticleRow],
    current_hierarchy: dict[LEVEL, Node],
) -> int | None:
    if current_article_number is None or not current_article_ends_clean(current_article_lines):
        return None

    return close_article(articles, current_article_number, current_article_lines, current_hierarchy)


def extract_articles_from_rows(rows: list[ArticleRow]) -> list[Article]:
    """Convert Docling document tables into Article records."""
    # Completed articles are appended here.
    articles: list[Article] = []

    # Current breadcrumb state: latest book/chapter/section/topic seen while scanning.
    hierarchy: dict[LEVEL, Node] = {}

    # Number of article currently being accumulated; None means not inside article yet.
    current_article_number: int | None = None

    # Expect next article number to be current_article_number + 1; None means no expectation yet.
    expected_next_article_number: int | None = None

    # Rows that belong to current article, including continuation rows.
    current_article_lines: list[ArticleRow] = []

    # Snapshot of hierarchy when current article started.
    current_hierarchy: dict[LEVEL, Node] = {}

    # Heading that Docling emitted without its title, e.g. "CHAPTER II".
    pending_heading: Node | None = None

    # Title-looking line promoted to TOPIC only if followed by an article.
    unclassified_node: Node | None = None

    # Read document row-by-row in natural Docling table order.
    for row in rows:
        # Check whether row is a book/chapter/section/topic/article marker.
        node = classify(row.text_en, row.text_ar)

        # Non-article headings update breadcrumb context but are not article text.
        # If line matching a heading start.
        if node and node.level != LEVEL.ARTICLE:
            # Close open article when heading mark is detected.
            # But make sure that the heading doesnt exist within an article
            # e.g. article line 1 \nSection 2 .... -> The articles mentions section 2.
            if current_article_number is not None and current_article_ends_clean(current_article_lines):
                next_after_repealed = close_article(
                    articles,
                    current_article_number,
                    current_article_lines,
                    current_hierarchy,
                )
                # If repealed articles notice found, use the article_number after its range
                if next_after_repealed is not None:
                    expected_next_article_number = next_after_repealed
                # Reset article state since it was closed
                current_article_number = None
                current_article_lines = []
                current_hierarchy = {}
            # Since we detected a heading, we need to update the hierarchy
            update_hierarchy(hierarchy, node)
            # In case actual heading title in the next line, dont close the heading yet.
            # The following loop for the heading completion is handled in the next block.
            # IMPORTANT: We assume that headings doesnt span more than one line,
            # so we close it directly if title text was also detected with the marker.
            pending_heading = node if not node.title_en else None
            # The node was already classified
            unclassified_node = None
            continue

        # Some headings arrive as two rows: marker row, then title row.
        # Make sure the current row has text and it is not an article. Otherwise we close the heading
        if pending_heading and row.text_en and not split_article_start(row.text_en, row.text_ar):
            # Now join the text title from last line with the current text title
            pending_heading.title_en = f"{pending_heading.title_en} {row.text_en.strip()}".strip()
            pending_heading.title_ar = f"{pending_heading.title_ar} {row.text_ar.strip()}".strip()

            # Close heading
            pending_heading = None
            continue

        # Any article/body row means pending heading title did not appear separately.
        # e.g. heading doesnt have title text, rather only marker.
        pending_heading = None

        # Create Repealed Articles
        # Before matching articles we check repealed articles standalone notices
        # otherwise the notice might get appended to the article.
        repealed = detect_repealed_range(f"{row.text_en} {row.text_ar}")
        # We make sure this text doesnt
        if repealed and current_article_number is None:
            start, end = repealed
            create_repealed_articles(articles, start, end, row.page_no, hierarchy)
            if end:
                expected_next_article_number = end + 1
            continue

        # Handle Article start mark.
        article_start = split_article_start(
            row.text_en,
            row.text_ar,
            expected_next_article_number,
            current_article_number,
        )

        if article_start:
            next_article_number, text_en, text_ar = article_start
            # This flag helps to ditinguish between real Article marks and
            # articles references within an article, since article references
            # look like an article start.
            is_expected = (
                expected_next_article_number is None or
                next_article_number == expected_next_article_number or
                (
                    current_article_number is not None and
                    next_article_number == current_article_number + 1
                )
            )

            # If last article not yet closed
            # curent here refers to the last article line
            if current_article_number is not None:
                # Check if current_article references other Articles
                # Most likely, the reference wouldnt point to the next article, at the same
                # time the current article line wouldn't have a clean ending.
                if not is_expected and not current_article_ends_clean(current_article_lines):
                    current_article_lines.append(row)
                    continue
                # Otherwise, this is an article ending.
                # Close last article from previous iteration and start new one.
                else:
                    next_article_after_repealed = close_article(
                        articles,
                        current_article_number,
                        current_article_lines,
                        current_hierarchy,
                    )
                    if next_article_after_repealed is not None:
                        expected_next_article_number = next_article_after_repealed
                    else:
                        # Set the expected next article number if no repealed notice detected
                        expected_next_article_number = next_article_number + 1

            # We dont set it in the closing article condition,
            # to accound for the first article
            current_article_number = next_article_number

            # Unnumbered topics appear before articles
            if unclassified_node is not None:
                unclassified_node.level = LEVEL.TOPIC
                update_hierarchy(hierarchy, unclassified_node)
                unclassified_node = None

            # Freeze current hierarchy for this article.
            # If hierarchy changed, we keep the article's hierarchy the same
            current_hierarchy = hierarchy.copy()
            # First row for new article keeps Arabic cell from same row.
            current_article_lines = [ArticleRow(text_en=text_en, text_ar=text_ar, page_no=row.page_no)]
            continue

        # Before handling the continuations of article lines,
        # we need to identify repealed ranges and unclassified nodes
        # The reason is that we need to identify unnumbered topics
        if (
            looks_like_title(row.text_en)
            and not row.text_en.casefold().startswith("articles ")
            and (LEVEL.TOPIC in hierarchy or LEVEL.SECTION in hierarchy)
        ):
            # Close open article, since the detected text is not an article continuation
            if current_article_number is not None and current_article_ends_clean(current_article_lines):
                next_article_after_repealed = close_article(
                    articles,
                    current_article_number,
                    current_article_lines,
                    current_hierarchy,
                )
                if next_article_after_repealed is not None:
                    expected_next_article_number = next_article_after_repealed
                else:
                    # Set the expected next article number if no repealed notice detected
                    expected_next_article_number = current_article_number + 1

                current_article_number = None
                current_article_lines = []
                current_hierarchy = {}

            # Unnumbered topics appear after headings
            if current_article_number is None:
                unclassified_node = Node(
                    level=LEVEL.TOPIC,
                    number="",
                    title_en=row.text_en.strip(),
                    title_ar=row.text_ar.strip(),
                )
                continue

        # If no new article starts, row is continuation of current article.
        if current_article_number is not None:
            current_article_lines.append(row)

    # End of document; close final open article.
    if current_article_number is not None:
        _ = close_article(
            articles,
            current_article_number,
            current_article_lines,
            current_hierarchy,
        )

    # Return extracted article records.
    return articles


def extract_articles(doc: DoclingDocument) -> list[Article]:
    return extract_articles_from_rows(iter_table_rows(doc))

def load_doc(path: Path) -> DoclingDocument:
    # Load Docling JSON exported earlier by convert.py.
    with path.open("r", encoding="utf-8") as fp:
        # Validate JSON dict back into DoclingDocument model.
        return DoclingDocument.model_validate(json.load(fp))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Load existing Docling JSON from project root.
    doc = load_doc(Path("law.pdf.json"))

    # Extract structured article records.
    articles = extract_articles_from_rows(iter_native_rows(doc))

    articles_dicts = [article.__dict__ for article in articles]

    # Log total count for quick sanity check.
    _log = logging.getLogger(__name__)
    _log.info("extracted %s articles", len(articles))
    _log.info(f"first article: {articles_dicts[0] if articles else 'none'}")

    with open("law.articles.json", "w", encoding="utf-8") as fp:
        fp.write(json.dumps(articles_dicts, ensure_ascii=False, indent=2))
