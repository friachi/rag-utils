import base64
import requests
from pathlib import Path


ENDPOINT = "http://localhost:8181/convert"

# tweak these to test overrides
MAX_EXAMPLE_LENGTH = 42
MAX_ENUM_VALUES = 7


def test_multipart(path: Path) -> None:
    print("=== multipart/form-data ===")

    params = {
        "max_example_length": MAX_EXAMPLE_LENGTH,
        "max_enum_values": MAX_ENUM_VALUES,
    }

    with path.open("rb") as f:
        files = {
            "file": (path.name, f, "application/octet-stream")
        }
        r = requests.post(ENDPOINT, params=params, files=files)

    r.raise_for_status()
    data = r.json()

    print("detected_type:", data["detected_type"])
    print("markdown preview:")
    print(data["markdown"][:500])
    print()


def test_json_base64(path: Path) -> None:
    print("=== json + base64 ===")

    raw = path.read_bytes()
    payload = {
        "filename": path.name,
        "content_base64": base64.b64encode(raw).decode("ascii"),
        # optional: can also pass overrides in body if you enabled that
        "max_example_length": MAX_EXAMPLE_LENGTH,
        "max_enum_values": MAX_ENUM_VALUES,
    }

    r = requests.post(
        ENDPOINT,
        params={
            # keep these too so you test precedence logic
            "max_example_length": MAX_EXAMPLE_LENGTH,
            "max_enum_values": MAX_ENUM_VALUES,
        },
        json=payload,
    )

    try:
     r.raise_for_status()
    except requests.HTTPError:
        print("status:", r.status_code)
        print("response:", r.text)
        raise
    data = r.json()

    print("detected_type:", data["detected_type"])
    print("markdown preview:")
    print(data["markdown"][:500])
    print()


if __name__ == "__main__":
    # change this to any file you want to test
    test_file = Path("/Users/fahed/github/rag-utils/docs/Anna/ISIN/Foreign_Exchange/Foreign_Exchange.Forward.Contract_For_Difference.InstRefDataReporting.V1.json")

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    test_multipart(test_file)
    test_json_base64(test_file)
