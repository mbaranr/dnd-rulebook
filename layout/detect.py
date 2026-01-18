from pathlib import Path
from typing import List, Dict

from paddleocr import LayoutDetection


class PaddleLayoutDetector:
    """
    Wrapper around PaddleOCR LayoutDetection.
    Detects high-level document regions (text, table, figure, title, etc.).
    """

    def __init__(self, min_confidence: float = 0.5):
        self.detector = LayoutDetection()
        self.min_confidence = min_confidence

    def detect_page(self, image_path: Path) -> List[Dict]:
        """
        Returns layout regions for a single page.
        """
        result = self.detector.predict(str(image_path))

        if not result or not result[0].get("boxes"):
            return []

        regions: List[Dict] = []

        for box in result[0]["boxes"]:
            if float(box["score"]) < self.min_confidence and \
               box["label"] not in {"footer", "number", "aside_text", "image"}:
                continue
            regions.append(
                {
                    "label": box["label"],
                    "confidence": float(box["score"]),
                    "bbox": [int(x) for x in box["coordinate"]],
                }
            )

        return regions