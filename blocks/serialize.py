from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

from utils.normalize import normalize_text
from blocks.assemble import Block


def flatten_toc(
    toc: List[Dict[str, Any]],
    path: List[str] | None = None,
) -> List[Tuple[str, int]]:
    """
    Flatten TOC tree into a list of (title, level) entries
    in document order.
    """
    path = path or []
    flat: List[Tuple[str, int]] = []

    for node in toc:
        title = node["title"]
        level = node["level"]
        flat.append((title, level))

        if node.get("children"):
            flat.extend(flatten_toc(node["children"], path + [title]))

    return flat

def normalize_title(text: str) -> str:
    """
    Normalize titles for matching:
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def titles_match(block_title: str, toc_title: str) -> bool:
    """
    Loose but deterministic title match.
    """
    bt = normalize_title(block_title)
    tt = normalize_title(toc_title)

    return bt == tt or bt in tt or tt in bt


def align_blocks_to_toc(
    blocks: List[Dict[str, Any]],
    toc: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Assign section_path to each block based on TOC alignment.
    """
    toc_entries = flatten_toc(toc)

    current_path: Dict[int, str] = {}
    block_to_path: Dict[str, List[str]] = {}

    for block in blocks:
        title = block.get("title")

        if title:
            for toc_title, level in toc_entries:
                if titles_match(title, toc_title):
                    # truncate deeper levels
                    current_path = {
                        k: v for k, v in current_path.items()
                        if k < level
                    }
                    current_path[level] = toc_title
                    break

        # build ordered section path
        section_path = [
            current_path[k]
            for k in sorted(current_path)
        ]

        block_to_path[block["block_id"]] = section_path

    return block_to_path


def block_to_dict(
    block: Block,
    section_path: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Convert a Block dataclass into a JSON-serializable dict.
    """
    data: Dict[str, Any] = {
        "block_id": block.block_id,
        "page": block.page,
        "type": block.type,
        "section_path": section_path or [],
    }

    if block.title:
        data["title"] = block.title

    if block.type == "text":
        data["text"] = normalize_text(block.text or "")

    if block.type == "table":
        data["image"] = block.image_crop

    return data


def serialize_blocks(
    blocks: List[Block],
    out_path: Path,
    toc: List[Dict[str, Any]] | None = None,
):
    """
    Serialize blocks into blocks.json and tables.json.
    """
    out_path.mkdir(parents=True, exist_ok=True)

    text_blocks: List[Dict] = []
    table_blocks: List[Dict] = []

    section_paths = None
    if toc is not None:
        section_paths = align_blocks_to_toc(blocks, toc)

    for block in blocks:
        section_path = None
        if section_paths is not None:
            section_path = section_paths.get(block.block_id)

        data = block_to_dict(
            block=block,
            section_path=section_path,
        )

        if block.type == "text":
            text_blocks.append(data)
        elif block.type == "table":
            table_blocks.append(data)

    # write text blocks
    blocks_path = out_dir / "blocks.json"
    with blocks_path.open("w", encoding="utf-8") as f:
        json.dump(text_blocks, f, indent=2, ensure_ascii=False)

    # write tables
    tables_path = out_dir / "tables.json"
    with tables_path.open("w", encoding="utf-8") as f:
        json.dump(table_blocks, f, indent=2, ensure_ascii=False)

    return {
        "blocks": blocks_path,
        "tables": tables_path,
        "num_text_blocks": len(text_blocks),
        "num_tables": len(table_blocks),
    }