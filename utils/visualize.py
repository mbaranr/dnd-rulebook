from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np


def visualize_ocr(
    image_path: Path,
    ocr_regions: List[Dict],
    reading_order: Optional[List[int]] = None,
    title: str = "OCR Reading Order",
):
    """
    Visualize OCR regions with optional reading order.
    Boxes are blue; order labels are large and red.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_plot = img.copy()

    boxes = [r["bbox"] for r in ocr_regions]

    # reading_order = [region_idx_at_pos_0, region_idx_at_pos_1, ...]
    order_map = None
    if reading_order is not None:
        order_map = {
            region_idx: order
            for order, region_idx in enumerate(reading_order)
        }

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(
            img_plot,
            (x1, y1),
            (x2, y2),
            (255, 0, 0), 
            2,
        )

        if order_map is None:
            continue

        order = order_map.get(i)
        if order is None:
            continue

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.putText(
            img_plot,
            str(order),
            (cx - 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,           
            (0, 0, 255),   
            3,             
            cv2.LINE_AA,
        )

    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()


def visualize_layout_detection(
    image_path: Path,
    layout_regions: List[Dict],
    title: str = "Layout Detection",
):
    """
    Visualize layout detection regions on a page image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_plot = img.copy()

    labels = sorted({r["label"] for r in layout_regions})
    cmap = colormaps.get_cmap("tab20")

    color_map = {
        label: tuple(int(c * 255) for c in cmap(i % 20)[::-1][:3])
        for i, label in enumerate(labels)
    }

    for r in layout_regions:
        x1, y1, x2, y2 = r["bbox"]
        color = color_map[r["label"]]

        pts = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=int,
        )
        cv2.polylines(img_plot, [pts], True, color, 2)

        label_text = f"{r['label']} ({r['confidence']:.2f})"
        cv2.putText(
            img_plot,
            label_text,
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()