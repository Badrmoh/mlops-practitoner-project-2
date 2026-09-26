import re
from dataclasses import dataclass
from statistics import median

from docling_core.types.doc.document import DoclingDocument
from legrag.ingestion.schemas import NativeTextCell, TextLine
from legrag.ingestion.normalize import normalize_arabic


AR_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def classify_cell_text(text: str) -> str:
    has_ar = AR_RE.search(text) is not None
    has_en = LATIN_RE.search(text) is not None

    if has_ar and has_en:
        return "mixed"
    if has_ar:
        return "ar"
    if has_en:
        return "en"
    return "unknown"


def extract_native_cells(doc: DoclingDocument) -> list[NativeTextCell]:
    cells: list[NativeTextCell] = []

    for page_no in sorted(int(page_no) for page_no in doc.pages):
        for item, _ in doc.iterate_items(page_no=page_no, traverse_pictures=False):
            text = getattr(item, "text", "").strip()
            provenance = getattr(item, "prov", None) or []
            if not text or not provenance:
                continue

            bbox = provenance[0].bbox
            cells.append(
                NativeTextCell(
                    text=text,
                    x0=bbox.l,
                    x1=bbox.r,
                    y=bbox.t,
                    page_no=page_no,
                )
            )

    return cells


def build_text_line(cells: list[NativeTextCell]) -> TextLine:
    classified = [(cell, classify_cell_text(cell.text)) for cell in cells]

    en_cells: list[NativeTextCell] = []
    ar_cells: list[NativeTextCell] = []
    page_no = cells[0].page_no

    for index, (cell, lang) in enumerate(classified):
        if lang == "en":
            en_cells.append(cell)
        elif lang == "ar":
            ar_cells.append(cell)
        else:
            if not en_cells and not ar_cells:
                continue
            if cell.text.strip() in {"-", "–", "—"}:
                continue

            nearest_en_x = max((cell.x1 for cell in en_cells), default=float("-inf"))
            nearest_ar_x = min((cell.x0 for cell in ar_cells), default=float("inf"))

            if (cell.x_center - nearest_en_x) < (nearest_ar_x - cell.x_center):
                en_cells.append(cell)
            else:
                ar_cells.append(cell)

    text_en = " ".join(cell.text for cell in sorted(en_cells, key=lambda cell: cell.x0))
    text_ar = " ".join(cell.text for cell in sorted(ar_cells, key=lambda cell: cell.x0, reverse=True))
    text_ar = normalize_arabic(text_ar)

    barrier_x = int(en_cells[-1].x1 + (ar_cells[0].x0 - en_cells[-1].x1) / 2) if en_cells and ar_cells else None

    return TextLine(
        en_cells=en_cells,
        ar_cells=ar_cells,
        barrier_x=barrier_x,
        text_en=text_en,
        text_ar=text_ar,
        normalized_en=text_en.lower().strip(),
        normalized_ar=text_ar.lower().strip(),
        page_no=page_no
    )


def group_cells_by_line(cells: list[NativeTextCell], y_tolerance: float = 3.0) -> list[TextLine]:
    """
    Groups a list of NativeTextCell objects into lines based on their y-coordinates.
    Cells are considered to be on the same line if they are on the same page
    and their y-coordinates are within the specified y_tolerance.
    """

    ordered = sorted(cells, key=lambda cell: (cell.page_no, -cell.y, cell.x0))
    text_lines: list[TextLine] = []
    current_line: list[NativeTextCell] = []
    current_page_no: int | None = None
    current_line_y: float = 0.0

    for cell in ordered:
        if current_line and cell.page_no == current_page_no and abs(cell.y - current_line_y) <= y_tolerance:
            current_line.append(cell)
        else:
            if current_line:
                text_lines.append(build_text_line(current_line))
            current_line_y = cell.y
            current_page_no = cell.page_no
            current_line = [cell]
    if current_line:
        text_lines.append(build_text_line(current_line))

    return text_lines
