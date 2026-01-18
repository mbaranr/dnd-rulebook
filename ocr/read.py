from pathlib import Path
from typing import Dict
import fitz 


class PyMuPDFPageOCR:
    """
    PyMuPDF OCR wrapper.
    Produces pixel-space boxes + text.
    """
    def __init__(self, lang: str = "eng", dpi: int = 300):
        self.lang = lang
        self.dpi = dpi
        self.scale = dpi / 72.0

    def ocr_page(self, pdf_path: Path, page_number: int) -> Dict:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number]

        # run ocr
        textpage = page.get_textpage_ocr(language=self.lang)
        data = textpage.extractDICT()

        regions = []

        for block in data["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    font_size = span["size"]

                    if not text:
                        continue

                    x0, y0, x1, y1 = span["bbox"]

                    # convert to pixel space
                    x0 *= self.scale
                    y0 *= self.scale
                    x1 *= self.scale
                    y1 *= self.scale

                    regions.append({
                        "text": text,
                        "size": font_size,
                        "bbox": [int(x0), int(y0), int(x1), int(y1)]
                    })

        return regions