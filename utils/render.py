from pathlib import Path
from typing import Iterator, Dict
import fitz
from PIL import Image


def render_pdf_pages(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
) -> Iterator[Dict]:
    """
    Render each page of a PDF to a PNG image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            image_path = out_dir / f"page_{page_index+1:04d}.png"

            if image_path.exists():
                yield {
                    "page_index": page_index,
                    "width": None,
                    "height": None,
                    "dpi": dpi,
                    "image_path": image_path,
                }
                continue

            page = doc[page_index]

            pix = page.get_pixmap(matrix=matrix, alpha=False)

            img = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples
            )

            img.save(image_path, format="PNG", optimize=True)

            yield {
                "page_index": page_index,
                "width": pix.width,
                "height": pix.height,
                "dpi": dpi,
                "image_path": image_path,
            }