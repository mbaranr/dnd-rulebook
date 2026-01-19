import json
from pathlib import Path
from typing import List, Dict

from utils.normalize import normalize_title


HEADING_DECORATOR_PREFIX = "<<<HEADING:"
HEADING_DECORATOR_SUFFIX = ">>>"


def split_into_sections(text_blocks: List[Dict], out_path: Path):
    sections: List[Dict] = []

    current = None
    section_index = 0

    for block in text_blocks:
        toc = tuple(block.get("toc", []))
        text = block.get("text", "").strip()
        page = block.get("page")
        block_id = block.get("block_id")
        title = block.get("title")

        if not text:
            continue

        toc_leaf = toc[-1] if toc else None

        # determine whether this block introduces a heading
        is_heading = (
            title
            and toc_leaf
            and normalize_title(title) != normalize_title(toc_leaf)
        )

        # inject heading into text if applicable
        if is_heading:
            decorated_heading = (
                f"{HEADING_DECORATOR_PREFIX} {title} {HEADING_DECORATOR_SUFFIX}"
            )
            text = decorated_heading + "\n\n" + text

        # start new section if TOC path changes
        if current is None or current["toc"] != list(toc):
            if current:
                sections.append(current)

            current = {
                "section_id": f"sec_{section_index:06d}",
                "toc": list(toc),
                "headings": [],
                "text": text,
                "pages": [page],
                "block_ids": [block_id],
            }
            section_index += 1
        else:
            # extend current section
            current["text"] += "\n\n" + text
            current["block_ids"].append(block_id)

            if page not in current["pages"]:
                current["pages"].append(page)

        # heading metadata tracking (unchanged semantics)
        if is_heading:
            if not current["headings"] or current["headings"][-1] != title:
                current["headings"].append(title)

    if current:
        sections.append(current)

    with (out_path / "text.json").open("w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    return sections