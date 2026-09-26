import json
import logging
from pathlib import Path
from typing import Tuple

from docling.datamodel.base_models import ConfidenceReport, ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.document_converter import DocumentConverter, NativePdfFormatOption
from docling_core.types.doc.page import TextCellUnit
from docling_core.types.doc.document import DoclingDocument


_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def convert(paths: list[Path]) -> Tuple[DoclingDocument, ConfidenceReport, ConversionStatus]:
    """
    Call Docling Native PDF pipeline.
    """

    pipeline_options = NativePdfPipelineOptions(
        text_cell_unit=TextCellUnit.WORD,
        generate_page_images=False,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result: ConversionResult

    if len(paths) == 1:
        # start_time = time.time()
        result = converter.convert(paths[0])
        # end_time = time.time() - start_time
    else:
        raise NotImplementedError("Batch conversion is not implemented")
    # _log.info(f"document converted in {end_time:.2f} seconds.")
    # _log.info(conv_result.confidence.mean_grade)
    # _log.info(conv_result.confidence.low_grade)

    return result.document, result.confidence, result.status


if __name__ == "__main__":
    pdf_path = Path("data/raw/law.pdf")
    doc, _, _ = convert([pdf_path])
    with Path(f"{pdf_path.name}.json").open("w") as fp:
        fp.write(json.dumps(doc.export_to_dict()))
