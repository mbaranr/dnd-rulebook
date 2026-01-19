import re
import unicodedata


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\s+([a-z])")

BULLET_RE = re.compile(r"[•◦∙·‣▪▫–—]")

MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def remove_control_chars(text: str) -> str:
    return CONTROL_CHARS_RE.sub("", text)

def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters into a consistent form.
    """
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "´": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }

    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    return text

def normalize_bullets(text: str) -> str:
    """
    Normalize bullet characters but do NOT remove them.
    """
    return BULLET_RE.sub("-", text)

def fix_hyphenated_linebreaks(text: str) -> str:
    """
    Merge hyphenated OCR breaks:
        cit- ies -> cities

    Only merges when the continuation starts with lowercase.
    """
    while True:
        new_text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
        if new_text == text:
            break
        text = new_text
    return text

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    """
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    if not text:
        return text

    text = remove_control_chars(text)
    text = normalize_unicode(text)
    text = normalize_bullets(text)
    text = fix_hyphenated_linebreaks(text)
    text = normalize_whitespace(text)

    return text

def normalize_title(text: str) -> str:
    """
    Normalize titles for matching.
    """
    text = text.lower()

    # normalize chapter variants
    text = re.sub(r"\bchapter\b", "ch", text)
    text = re.sub(r"\bch\.\b", "ch", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()