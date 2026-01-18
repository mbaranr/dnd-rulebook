from typing import List


def intersects(b1: List[int], b2: List[int], min_overlap: float = 0.3):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if x2 <= x1 or y2 <= y1:
        return False

    inter = (x2 - x1) * (y2 - y1)
    area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    return inter / max(area, 1) >= min_overlap