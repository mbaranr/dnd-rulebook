import json
import re
import fitz
import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


TOC_LINE_RE = re.compile(r"^(?P<title>.+?)\.{2,}\s*(?P<page>\d+)$")

TOC_NOISE = {
    "CONTENTS",
    "TABLE OF CONTENTS",
}

class TOCEntry(BaseModel):
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    level: int
    children: List["TOCEntry"] = Field(default_factory=list)

TOCEntry.model_rebuild()


def is_toc_noise(text: str, font_size: float):
    t = text.strip().upper()

    if t in TOC_NOISE:
        return True

    # page numbers alone
    if t.isdigit():
        return True

    # footers
    if font_size <= 7:
        return True

    return False

def extract_toc_lines(pdf_path: str, toc_page: int = 1):
    with fitz.open(pdf_path) as doc:
        page = doc[toc_page]
        lines = []

        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line["spans"]
                text = "".join(span["text"] for span in spans).strip()
                if not text:
                    continue

                lines.append({
                    "text": text,
                    "font_size": max(span["size"] for span in spans),
                    "x0": min(span["bbox"][0] for span in spans),
                    "y0": min(span["bbox"][1] for span in spans),
                })

        return lines

def parse_toc_lines(lines):
    # filter noise
    clean = [
        l for l in lines
        if not is_toc_noise(l["text"], l["font_size"])
    ]

    # derive hierarchy from font size
    font_sizes = sorted({l["font_size"] for l in clean}, reverse=True)
    font_to_level = {fs: lvl for lvl, fs in enumerate(font_sizes)}

    entries = []

    for l in clean:
        text = l["text"]
        level = font_to_level[l["font_size"]]

        match = TOC_LINE_RE.match(text)
        if match:
            title = match.group("title").strip()
            page_start = int(match.group("page"))
        else:
            title = text
            page_start = None

        entries.append({
            "title": title,
            "page_start": page_start,
            "level": level,
        })

    return entries

def build_toc_tree(entries):
    stack: List[TOCEntry] = []
    roots: List[TOCEntry] = []

    for e in entries:
        node = TOCEntry(
            title=e["title"],
            page_start=e["page_start"],
            level=e["level"],
        )

        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)

        stack.append(node)

    return roots

def compute_page_ranges(nodes: list[TOCEntry], last_page: int):

    def start(nodes: list[TOCEntry]):
        for node in nodes:
            if node.children:
                start(node.children)
                if node.page_start is None:
                    node.page_start = node.children[0].page_start

    def end(siblings: list[TOCEntry], inherited_end: int):
        for i, node in enumerate(siblings):
            next_sibling = siblings[i + 1] if i + 1 < len(siblings) else None

            if next_sibling:
                if next_sibling.page_start > node.page_start:
                    node.page_end = next_sibling.page_start - 1
                else:
                    node.page_end = node.page_start
            else:
                node.page_end = inherited_end

            if node.children:
                end(node.children, node.page_end)

    start(nodes)
    end(nodes, last_page)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    with (repo_root / "config" / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)

    pdf_path = repo_root / cfg["ingest"]["paths"]["pdf"]
    out_path = repo_root / cfg["ingest"]["paths"]["toc"]

    lines = extract_toc_lines(pdf_path)
    entries = parse_toc_lines(lines)
    toc_tree = build_toc_tree(entries)
    
    print(toc_tree)

    with fitz.open(pdf_path) as doc:
        compute_page_ranges(toc_tree, doc.page_count)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            [n.model_dump() for n in toc_tree],
            f,
            indent=2,
            ensure_ascii=False
        )