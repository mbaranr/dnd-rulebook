import json
import base64
import time
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError


BASE_MAX_TOKENS = 1024
MAX_RETRIES = 3

# throttling
SUCCESS_SLEEP_SECONDS = 4
RETRY_BASE_SLEEP_SECONDS = 3


SYSTEM_PROMPT = """
You are a vision-language model that extracts structured tables from images.

You MUST return valid JSON only.
No markdown.
No explanations.
No extra text.

Rules:
- Extract the table exactly as shown.
- If column headers are missing or unclear, infer them from:
  - table layout
  - surrounding context
  - toc and title
- Normalize all values as strings.
- Rows must be an array of objects.
- Column names must be consistent across rows.
- If the table is multi-row but header appears mid-table, normalize it.
- Always produce a short natural-language description of the table.
"""


def encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_prompt(block: Dict) -> str:
    toc = " > ".join(block.get("toc", []))
    title = block.get("title") or "None"

    return f"""
Context:
TOC path: {toc}
Section title: {title}

Task:
Extract the table from the image and return a JSON object with this schema:

{{
  "columns": [string],
  "rows": [
    {{ column_name: value, ... }}
  ],
  "description": string
}}

The description should briefly explain:
- what the table represents
- what the rows correspond to
- what the columns mean
""".strip()


def extract_table(
    client: InferenceClient,
    block: Dict
) -> Dict[str, Any]:

    image_b64 = encode_image(Path(block["image_crop"]))
    prompt = build_prompt(block)

    # progressive token schedule (capped)
    token_schedule = [
        BASE_MAX_TOKENS,
        BASE_MAX_TOKENS * 2,
        BASE_MAX_TOKENS * 4,
    ]

    last_error: Exception | None = None
    last_content: str | None = None

    for attempt, max_tokens in enumerate(token_schedule, start=1):
        print(
            f"[VLM] Block {block['block_id']} – "
            f"attempt {attempt}/{len(token_schedule)} "
            f"(max_tokens={max_tokens})"
        )

        try:
            response = client.chat.completions.create(
                model=client.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )

        except HfHubHTTPError as e:
            # explicit rate-limit handling
            if e.response is not None and e.response.status_code == 429:
                sleep_time = RETRY_BASE_SLEEP_SECONDS * attempt
                print(f"[VLM] 429 rate limit hit. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                last_error = e
                continue
            raise

        content = response.choices[0].message.content
        last_content = content

        try:
            table = json.loads(content)

            # mandatory throttle on success
            time.sleep(SUCCESS_SLEEP_SECONDS)

            return {
                "table_id": block['block_id'],
                "toc": block.get("toc", []),
                "title": block.get("title"),
                "page": block.get("page"),
                "description": table.get("description"),
                "columns": table.get("columns", []),
                "rows": table.get("rows", []),
                "image_crop": block.get("image_crop"),
            }

        except json.JSONDecodeError as e:
            print(f"[VLM] JSON parse error on attempt {attempt}: {e}")
            last_error = e
            time.sleep(RETRY_BASE_SLEEP_SECONDS * attempt)

    # all attempts failed
    raise RuntimeError(
        f"Invalid JSON after {len(token_schedule)} attempts "
        f"for block {block['block_id']}:\n{last_content}"
    ) from last_error