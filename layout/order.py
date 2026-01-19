from collections import defaultdict
from typing import List, Dict

import torch
from transformers import LayoutLMv3ForTokenClassification


MAX_LEN = 510
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2


def boxes2inputs(boxes: List[List[int]]):
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1] + [1] * len(boxes) + [1]
    return {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }

def prepare_inputs(inputs: Dict[str, torch.Tensor], model: LayoutLMv3ForTokenClassification):
    ret = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        ret[k] = v
    return ret

def parse_logits(logits: torch.Tensor, length: int):
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]

    while True:
        collisions = defaultdict(list)
        for idx, order in enumerate(ret):
            collisions[order].append(idx)

        collisions = {k: v for k, v in collisions.items() if len(v) > 1}
        if not collisions:
            break

        for order, idxes in collisions.items():
            scored = {
                idx: logits[idx, order].item()
                for idx in idxes
            }
            ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
            for idx, _ in ranked[1:]:
                ret[idx] = orders[idx].pop()

    return ret


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
    def get_reading_order(self, ocr_regions: List[Dict]):
        max_x = max(r["bbox"][2] for r in ocr_regions)
        max_y = max(r["bbox"][3] for r in ocr_regions)

        boxes = []
        for r in ocr_regions:
            x0, y0, x1, y1 = r["bbox"]

            boxes.append([
                int((x0 / max_x) * 1000),
                int((y0 / max_y) * 1000),
                int((x1 / max_x) * 1000),
                int((y1 / max_y) * 1000),
            ])

        inputs = boxes2inputs(boxes)
        inputs = prepare_inputs(inputs, self.model)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0).cpu()

        return parse_logits(logits, len(boxes))


def heuristic_reading_order(ocr_regions: List[Dict], num_columns: int = 2):
    """
    Deterministic reading order.

    Rules:
    - Split page width into `num_columns`
    - Process columns left -> right
    - Within each column: top -> bottom and left -> right
    """

    if num_columns < 1:
        raise ValueError("num_columns must be >= 1")

    # determine page width
    page_width = max(r["bbox"][2] for r in ocr_regions)
    col_width = page_width / num_columns

    indexed = list(enumerate(ocr_regions))

    def column_index(region):
        x0, _, x1, _ = region["bbox"]
        center_x = (x0 + x1) / 2
        return min(int(center_x // col_width), num_columns - 1)

    # assign column
    buckets = defaultdict(list)
    for idx, r in indexed:
        col = column_index(r)
        buckets[col].append((idx, r))

    ordered_indices: List[int] = []

    for col in range(num_columns):
        col_items = buckets.get(col, [])

        # sort top-to-bottom, then left-to-right
        col_items.sort(
            key=lambda x: (
                x[1]["bbox"][1],  # y0
                x[1]["bbox"][0],  # x0
            )
        )

        ordered_indices.extend(idx for idx, _ in col_items)

    return ordered_indices