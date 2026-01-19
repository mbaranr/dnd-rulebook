from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from utils.normalize import normalize_text, normalize_title


def flatten_toc(toc: List[Dict[str, Any]]):
    """
    Flatten hierarchical TOC into a linear list, preserving paths.
    """
    flat: List[Dict[str, Any]] = []

    def walk(nodes: List[Dict[str, Any]], path: List[str]):
        for n in nodes:
            current_path = path + [n["title"]]
            flat.append(
                {
                    "title": n["title"],
                    "page_start": n.get("page_start"),
                    "level": n.get("level"),
                    "path": current_path,
                }
            )
            if n.get("children"):
                walk(n["children"], current_path)

    walk(toc, [])
    return flat

def match_toc_entry(block, toc_entries):
    """
    Strict title match + page sanity check.
    Returns the matched TOC entry or None.
    """
    block_title = block.get("title")
    block_page = block.get("page")

    if not block_title:
        return None

    bt = normalize_title(block_title)

    for entry in toc_entries:
        if normalize_title(entry["title"]) != bt:
            continue

        toc_page = entry.get("page_start")
        if toc_page is not None and block_page is not None:
            if block_page < toc_page:
                continue

        return entry

    return None


def serialize_blocks(
    blocks: List[Dict[str, Any]],
    out_path: Path,
    toc: Optional[List[Dict[str, Any]]] = None,
    pages_to_skip: Optional[List[int]] = None,
):
    """
    Serialize blocks into text_blocks.json and table_blocks.json.
    """

    out_path.mkdir(parents=True, exist_ok=True)
    pages_to_skip = set(pages_to_skip or [])

    toc_entries: List[Dict[str, Any]] = []
    if toc:
        toc_entries = flatten_toc(toc)

    # global ordering first (page-local order_id)
    blocks = sorted(blocks, key=lambda b: (b["page"], b["order_id"]))

    text_blocks: List[Dict[str, Any]] = []
    table_blocks: List[Dict[str, Any]] = []

    current_toc_path: List[str] = []
    last_text_block_with_title: Optional[Dict[str, Any]] = None

    for block in blocks:
        page = block.get("page")
        if page in pages_to_skip:
            continue

        # toc state update
        if toc_entries:
            matched_entry = match_toc_entry(block, toc_entries)
            if matched_entry:
                current_toc_path = matched_entry["path"]

        # text blocks
        if block["type"] == "text":
            text = normalize_text(block.get("text", ""))

            if not block.get("title") and last_text_block_with_title:
                # merge into previous titled block
                last_text_block_with_title["text"] += "\n\n" + text
                continue

            block_out = {
                "block_id": block["block_id"],
                "page": page,
                "order_id": block["order_id"],
                "title": block.get("title"),
                "toc": current_toc_path.copy(),
                "text": text,
            }

            text_blocks.append(block_out)

            if block.get("title"):
                last_text_block_with_title = block_out

        # table blocks
        elif block["type"] == "table":
            block_out = {
                "block_id": block["block_id"],
                "page": page,
                "order_id": block["order_id"],
                "title": block.get("title"),
                "toc": current_toc_path.copy(),
                "image_crop": block.get("image_crop"),
            }
            table_blocks.append(block_out)

    with (out_path / "text_blocks.json").open("w", encoding="utf-8") as f:
        json.dump(text_blocks, f, indent=2, ensure_ascii=False)

    with (out_path / "table_blocks.json").open("w", encoding="utf-8") as f:
        json.dump(table_blocks, f, indent=2, ensure_ascii=False)

    return text_blocks, table_blocks