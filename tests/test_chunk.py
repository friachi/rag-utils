"""
test_post_chunk.py

Simple client script to test POST /chunk endpoint.

Requirements:
  pip install requests

Run:
  python test_post_chunk.py
"""

import json
import requests

URL = "http://127.0.0.1:8181/chunk"


MARKDOWN_SAMPLE = """
# Swap
Intro text.

## Leg
General leg description.

### FixedRate
- rate: number
- dayCount: string
- paymentFrequency: string

### FloatingRate
- index: string
- spread: number

## Settlement
- currency: string
- settlementDate: date
"""


def main() -> None:
    payload = {
        "markdown": MARKDOWN_SAMPLE,
        # try 0 to exercise auto-selection
        "max_header_level": 3,

        # everything else optional (defaults are sensible)
        "max_chars": 1200,
        "recursive_chunk_size": 600,
        "recursive_overlap": 80,
        "include_path_in_text": True,
        "source_id": "example://swap-schema",
    }

    print("POST", URL)
    resp = requests.post(URL, json=payload, timeout=30)

    print("Status:", resp.status_code)
    resp.raise_for_status()

    data = resp.json()

    print("\nChosen max_header_level:", data["chosen_max_header_level"])
    print("Number of chunks:", len(data["chunks"]))

    # Print a preview of each chunk
    for i, ch in enumerate(data["chunks"]):
        print("\n" + "=" * 80)
        print(f"Chunk {i}")
        print("Header path:", ch["metadata"].get("header_path"))
        print("Is subchunk:", ch["metadata"].get("is_subchunk", False))
        print("-" * 80)
        print(ch["text"][:500].rstrip())
        if len(ch["text"]) > 500:
            print("... (truncated)")

    # Optional: pretty-print full JSON for debugging
    # print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
