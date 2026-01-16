from pathlib import Path
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np


def visualize_layout_detection(
    image_path: Path,
    layout_regions: List[Dict],
    min_confidence: float = 0.5,
    title: str = "Layout Detection",
):
    img = cv2.imread(str(image_path))
    img_plot = img.copy()

    labels = sorted({r["label"] for r in layout_regions})
    cmap = colormaps.get_cmap("tab20")

    color_map = {
        label: tuple(int(c * 255) for c in cmap(i % 20)[::-1][:3])
        for i, label in enumerate(labels)
    }

    for r in layout_regions:
        if r["confidence"] < min_confidence:
            continue

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