from pathlib import Path
from typing import Dict, List
import json

from paddleocr import PaddleOCR
from PIL import Image


class PaddlePageOCR:
    """
    PaddleOCR wrapper using the predict() API.
    Produces LayoutLM-compatible words + boxes.
    """

    def __init__(self, lang: str = "en"):
        self.ocr = PaddleOCR(
            lang=lang,
        )

    def ocr_page(self, image_path: Path) -> Dict:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        result = self.ocr.predict(str(image_path))
        if not result:
            return {
                "image": str(image_path),
                "width": width,
                "height": height,
                "words": [],
                "boxes": [],
                "scores": [],
            }

        page = result[0]

        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys = page.get("rec_polys", [])

        words: List[str] = []
        boxes: List[List[int]] = []
        confs: List[float] = []

        for text, score, poly in zip(texts, scores, polys):
            if not isinstance(text, str) or not text.strip():
                continue

            # poly = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]

            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            # normalize to LayoutLM 0–1000 space
            norm_box = [
                int(x0 / width * 1000),
                int(y0 / height * 1000),
                int(x1 / width * 1000),
                int(y1 / height * 1000),
            ]

            words.append(text)
            boxes.append(norm_box)
            confs.append(float(score))

        return {
            "image": str(image_path),
            "width": width,
            "height": height,
            "words": words,
            "boxes": boxes,
            "scores": confs,
        }


def ocr_document_pages(
    image_dir: Path,
    out_dir: Path,
    lang: str = "en",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ocr = PaddlePageOCR(lang=lang)

    for image_path in sorted(image_dir.glob("page_*.png")):
        page_id = image_path.stem
        data = ocr.ocr_page(image_path)

        out_path = out_dir / f"{page_id}.ocr.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)