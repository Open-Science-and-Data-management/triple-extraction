"""REBEL: parse decoder output + (ต่อใน Task 4) sentence split / batch / infer

รูปแบบ output จริงของ rebel-large: `<trip> head <subj> relation <obj> tail`
(spec เขียนเป็น `<sep>` — parser รับทั้งสองรูปแบบ)
"""

import re
from dataclasses import dataclass, field

_SEP = re.compile(r"<subj>|<obj>|<sep>")


@dataclass
class ParsedTriples:
    triples: list[dict] = field(default_factory=list)
    # แถวที่ parse ไม่ได้เก็บ raw text ไว้ debug ไม่ทิ้งเงียบ ๆ
    unparsed: list[str] = field(default_factory=list)


def parse_rebel_output(raw: str) -> ParsedTriples:
    parsed = ParsedTriples()
    parts = raw.split("<trip>")
    # ชิ้นแรกคือข้อความก่อน tag แรก — ถ้ามีเนื้อจริงแปลว่า output เพี้ยน ทั้งชิ้นเก็บไว้
    if parts[0].strip():
        parsed.unparsed.append(parts[0])

    for segment in parts[1:]:
        cols = [c.strip() for c in _SEP.split(segment)]
        if len(cols) == 3 and all(cols):
            parsed.triples.append(
                {"head": cols[0], "relation": cols[1], "tail": cols[2]}
            )
        else:
            parsed.unparsed.append(segment.strip())
    return parsed
