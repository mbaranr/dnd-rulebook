import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from ocr.read import PyMuPDFPageOCR
from layout.detect import PaddleLayoutDetector
from layout.order import heuristic_reading_order
from utils.geometry import intersects


TOC_LINE_RE = re.compile(r"^(?P<title>.+?)\.{2,}\s*(?P<page>\d+)$")
DOT_LEADER_RE = re.compile(r"\.{2,}\s*$")

# ---------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------

def parse_toc_page(
    pdf_path: Path,
    img_path: Path,
    toc_page: int,
    num_columns: int,
) -> List[Dict]:
    """
    Extract raw TOC entries from a single TOC page.
    """

    ocr = PyMuPDFPageOCR()
    detector = PaddleLayoutDetector(min_confidence=0.8)

    ocr_regions = ocr.ocr_page(pdf_path, toc_page - 1)
    layout_regions = detector.detect_page(img_path)

    reading_order = heuristic_reading_order(
        ocr_regions,
        num_columns=num_columns,
    )

    ordered_ocr = [ocr_regions[i] for i in reading_order]

    # Keep only OCR inside detected TOC/content area
    filtered_ocr = []
    for ocr_region in ordered_ocr:
        for lr in layout_regions:
            if lr.get("label") == "content" and intersects(
                ocr_region["bbox"], lr["bbox"]
            ):
                filtered_ocr.append(ocr_region)
                break

    # Font size → hierarchy level
    font_sizes = sorted(
        {ocr["size"] for ocr in filtered_ocr},
        reverse=True,
    )
    font_to_level = {fs: lvl for lvl, fs in enumerate(font_sizes)}

    entries: List[Dict] = []

    for ocr in filtered_ocr:
        raw_text = ocr["text"].strip()
        text = DOT_LEADER_RE.sub("", raw_text).rstrip()
        
        level = font_to_level[ocr["size"]]

        match = TOC_LINE_RE.match(text)
        if match:
            title = match.group("title").strip()
            page_start = int(match.group("page"))
        else:
            title = text
            page_start = None

        entries.append(
            {
                "title": title,
                "page_start": page_start,
                "level": level,
            }
        )

    return entries


# ---------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------

def build_toc_tree(entries: List[Dict]) -> List[Dict]:
    """
    Convert flat TOC entries into a hierarchical tree.
    """

    stack: List[Dict] = []
    roots: List[Dict] = []

    for e in entries:
        node = {
            "title": e["title"],
            "page_start": e["page_start"],
            "page_end": None,
            "level": e["level"],
            "children": [],
        }

        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)

        stack.append(node)

    return roots


# ---------------------------------------------------------------------
# Page range resolution
# ---------------------------------------------------------------------

def compute_page_ranges(
    nodes: List[Dict],
    last_page: int,
) -> None:
    """
    Mutates the tree in place.
    Rule:
    - If a node has no page_start, duplicate it from its immediate child.
    """

    def fill_missing_starts(items: List[Dict]):
        for node in items:
            if node["children"]:
                fill_missing_starts(node["children"])
                if node["page_start"] is None:
                    # duplicate from immediate first child
                    node["page_start"] = node["children"][0]["page_start"]

    def assign_ends(siblings: List[Dict], inherited_end: int):
        for i, node in enumerate(siblings):
            next_node = siblings[i + 1] if i + 1 < len(siblings) else None

            if node["page_start"] is None:
                # should only happen for leaf nodes with no page info
                node["page_end"] = inherited_end
            elif next_node and next_node["page_start"] is not None:
                node["page_end"] = max(
                    node["page_start"],
                    next_node["page_start"] - 1,
                )
            else:
                node["page_end"] = inherited_end

            if node["children"]:
                assign_ends(node["children"], node["page_end"])

    fill_missing_starts(nodes)
    assign_ends(nodes, last_page)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def extract_toc(
    pdf_path: Path,
    img_path: Path,
    toc_page: int = 2,
    num_columns: int = 2,
    last_page: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Public TOC extraction API.
    """

    entries = parse_toc_page(
        pdf_path=pdf_path,
        img_path=img_path,
        toc_page=toc_page,
        num_columns=num_columns,
    )

    tree = build_toc_tree(entries)

    if last_page is not None:
        compute_page_ranges(tree, last_page)

    if out_path is not None:
        out_path = out_path / "toc.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

    return tree