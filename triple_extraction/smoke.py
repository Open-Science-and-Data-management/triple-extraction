"""CLI ทดสอบ pipeline ปลายทาง: python -m triple_extraction.smoke "Some text..."

โหลดโมเดลจริง (lazy ตาม extractor) พิมพ์ device แล้วคืน triples เป็น JSON
"""

import json
import sys
import time

from triple_extraction.extractor.rebel import RebelExtractor


def main() -> None:
    text = " ".join(sys.argv[1:])
    if not text.strip():
        sys.exit("usage: python -m triple_extraction.smoke <text>")

    ex = RebelExtractor()
    print(f"device: {ex.device}", flush=True)

    t0 = time.perf_counter()
    triples = ex.extract(text)
    elapsed = time.perf_counter() - t0

    print(json.dumps({"timing": {"total_ms": round(elapsed * 1000)},
                      "triples": triples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
