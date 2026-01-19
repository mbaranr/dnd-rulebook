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


def parse_toc_page(
    pdf_path: Path,
    img_path: Path,
    toc_page: int,
    num_columns: int
):
    """
    Extract raw TOC entries from a single TOC page.
    Page numbers are used only as sanity guards.
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

    filtered_ocr: List[Dict] = []
    for ocr_region in ordered_ocr:
        for lr in layout_regions:
            if lr.get("label") == "content" and intersects(
                ocr_region["bbox"], lr["bbox"]
            ):
                filtered_ocr.append(ocr_region)
                break

    # font size defines hierarchy
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

        # isolated page number line
        if text.isdigit():
            if entries and entries[-1]["page_start"] is None:
                entries[-1]["page_start"] = int(text)
            continue

        # normal toc line
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

def build_toc_tree(entries: List[Dict]):
    """
    Convert flat TOC entries into a hierarchical tree.
    """

    stack: List[Dict] = []
    roots: List[Dict] = []

    for e in entries:
        node = {
            "title": e["title"],
            "page_start": e["page_start"],
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

def propagate_page_starts(nodes: List[Dict]):
    """
    Ensure every structural node has a page_start by inheriting
    from its immediate first child when missing.
    """

    for node in nodes:
        if node["children"]:
            propagate_page_starts(node["children"])
            if node["page_start"] is None:
                node["page_start"] = node["children"][0]["page_start"]


def extract_toc(
    pdf_path: Path,
    img_path: Path,
    toc_page: int = 2,
    num_columns: int = 2,
    out_path: Optional[Path] = None
):
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
    propagate_page_starts(tree)

    if out_path is not None:
        out_path = out_path / "toc.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

    return tree