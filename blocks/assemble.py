from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from PIL import Image

from utils.geometry import intersects


@dataclass
class Block:
    block_id: str
    order_id: int
    page: int
    type: str               # "text" | "table"
    text: Optional[str] = None
    title: Optional[str] = None
    image_crop: Optional[str] = None


def crop_region(
    image_path: Path,
    bbox: List[int],
    out_path: Path,
    padding: int = 8,
):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = img.crop((x1, y1, x2, y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, format="PNG")


class BlockAssembler:
    def __init__(self, crop_dir: Path, prose_font_size: Optional[float] = None):
        self.crop_dir = crop_dir
        self.prose_font_size = prose_font_size

    def assemble_page(
        self,
        page_index: int,
        image_path: Path,
        ocr_regions: List[Dict],
        reading_order: List[int],
        layout_regions: List[Dict],
    ):
        blocks: List[Block] = []
        order_id = 0

        ordered_ocr = [ocr_regions[i] for i in reading_order]
        num_ocr = len(ordered_ocr)

        table_regions = [
            lr for lr in layout_regions if lr.get("label") == "table"
        ]
        emitted_tables: set[int] = set()

        title_state: Optional[str] = None
        current_text: List[str] = []

        def flush_text():
            nonlocal order_id
            if not current_text:
                return

            blocks.append(
                Block(
                    block_id=f"blk_{page_index + 1}_{order_id:05d}",
                    order_id=order_id,
                    page=page_index + 1,
                    type="text",
                    title=title_state,
                    text=" ".join(current_text),
                )
            )
            order_id += 1
            current_text.clear()

        for idx, ocr in enumerate(ordered_ocr):
            bbox = ocr["bbox"]
            text = ocr["text"]
            size = ocr.get("size")

            matched_layout = None
            for lr in layout_regions:
                if intersects(bbox, lr["bbox"]):
                    matched_layout = lr
                    break

            label = matched_layout["label"] if matched_layout else None

            # noise
            if label in {"footer", "number", "image", "aside_text"}:
                continue

            # explicit titles
            if label in {"paragraph_title", "figure_title"}:
                flush_text()
                title_state = text
                continue

            # tables
            table_index = None
            for i, tbl in enumerate(table_regions):
                if intersects(bbox, tbl["bbox"]):
                    table_index = i
                    break

            if table_index is not None:
                if table_index not in emitted_tables:
                    flush_text()

                    tbl = table_regions[table_index]
                    block_id = f"tbl_{page_index + 1}_{order_id:05d}"
                    crop_path = self.crop_dir / f"{block_id}.png"

                    crop_region(image_path, tbl["bbox"], crop_path)

                    blocks.append(
                        Block(
                            block_id=block_id,
                            order_id=order_id,
                            page=page_index + 1,
                            type="table",
                            title=title_state,
                            image_crop=str(crop_path),
                        )
                    )

                    emitted_tables.add(table_index)
                    order_id += 1

                continue

            # font size based heuristics
            if self.prose_font_size and size is not None:
                # smaller or equal to prose → always prose
                if size <= self.prose_font_size:
                    current_text.append(text)
                    continue

                prev_size = (
                    ordered_ocr[idx - 1].get("size")
                    if idx > 0 else None
                )
                next_size = (
                    ordered_ocr[idx + 1].get("size")
                    if idx < num_ocr - 1 else None
                )

                if prev_size != size and next_size != size:
                    flush_text()
                    title_state = text
                    continue

                # same-size neighbors, prose
                current_text.append(text)
                continue

            # default: prose
            current_text.append(text)

        flush_text()
        return blocks