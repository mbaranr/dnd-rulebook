from typing import List
import torch
from transformers import LayoutLMv3ForTokenClassification
from layoutreader.v3.helpers import boxes2inputs, prepare_inputs, parse_logits


class LayoutReader:
    def __init__(self, model_name="hantian/layoutreader", device=None):
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_reading_order(self, boxes_xyxy: List[List[int]]) -> List[int]:
        """
        boxes_xyxy: list of [x1,y1,x2,y2] in pixel space
        returns: reading order index per box
        """

        # normalize to 0–1000
        max_x = max(b[2] for b in boxes_xyxy)
        max_y = max(b[3] for b in boxes_xyxy)

        boxes_norm = [
            [
                int(b[0] / max_x * 1000),
                int(b[1] / max_y * 1000),
                int(b[2] / max_x * 1000),
                int(b[3] / max_y * 1000),
            ]
            for b in boxes_xyxy
        ]

        inputs = boxes2inputs(boxes_norm)
        inputs = prepare_inputs(inputs, self.model)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits.cpu().squeeze(0)
        return parse_logits(logits, len(boxes_norm))