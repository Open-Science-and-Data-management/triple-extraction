"""REBEL: parse decoder output + sentence split / batch / GPU infer

รูปแบบ output จริงของ rebel-large: `<trip> head <subj> relation <obj> tail`
(spec เขียนเป็น `<sep>` — parser รับทั้งสองรูปแบบ)
"""

import logging
import re
from dataclasses import dataclass, field

import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from triple_extraction.settings import MODEL_NAME

_TRIPLET = re.compile(r"<triplet>|<trip>")
# รูปแบบจริงของ rebel-large: <triplet> head <subj> tail <obj> relation
_REBEL_COLS = re.compile(r"<subj>|<obj>")
# รูปแบบใน spec: <trip> head <sep> relation <sep> tail
_SPEC_COLS = re.compile(r"<sep>")
# special tokens ของ tokenizer ที่ไม่เกี่ยวกับโครงสร้าง triple
_NOISE = re.compile(r"</?s>|<pad>")


@dataclass
class ParsedTriples:
    triples: list[dict] = field(default_factory=list)
    # แถวที่ parse ไม่ได้เก็บ raw text ไว้ debug ไม่ทิ้งเงียบ ๆ
    unparsed: list[str] = field(default_factory=list)


def parse_rebel_output(raw: str) -> ParsedTriples:
    parsed = ParsedTriples()
    raw = _NOISE.sub("", raw)
    parts = _TRIPLET.split(raw)
    # ชิ้นแรกคือข้อความก่อน tag แรก — ถ้ามีเนื้อจริงแปลว่า output เพี้ยน ทั้งชิ้นเก็บไว้
    if parts[0].strip():
        parsed.unparsed.append(parts[0])

    for segment in parts[1:]:
        if "<sep>" in segment:  # รูปแบบ spec: h <sep> r <sep> t
            cols = [c.strip() for c in _SPEC_COLS.split(segment)]
        else:  # รูปแบบจริง: h <subj> t <obj> r
            cols = [c.strip() for c in _REBEL_COLS.split(segment)]
        if len(cols) == 3 and all(cols):
            if "<sep>" in segment:
                head, relation, tail = cols
            else:
                head, tail, relation = cols
            parsed.triples.append(
                {"head": head, "relation": relation, "tail": tail}
            )
        else:
            parsed.unparsed.append(segment.strip())
    return parsed


class RebelExtractor:
    """แบ่งประโยคด้วย spaCy → batch ยิงโมเดลบน GPU (cuda default) → parse เป็น triples"""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None
        self._nlp = None

    def _ensure_loaded(self) -> None:
        # lazy-load: ไม่โหลดโมเดลตอน import/สร้าง instance — test ไม่พัง, startup ไม่ค้าง
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
                self.device
            )
            self._model.eval()
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> list[dict]:
        self._ensure_loaded()
        sents = list(self._nlp(text).sents)

        triples: list[dict] = []
        for i in range(0, len(sents), self.batch_size):
            batch = sents[i : i + self.batch_size]
            decoded = self._generate_batch([s.text for s in batch])
            for sentence_index, (sent, raw) in enumerate(zip(batch, decoded), start=i):
                parsed = parse_rebel_output(raw)
                if parsed.unparsed:
                    logging.debug("unparsed decoder output: %r", parsed.unparsed)
                for t in parsed.triples:
                    t.update(
                        sentence_index=sentence_index,
                        start=sent.start_char,
                        end=sent.end_char,
                        extractor="rebel",
                    )
                    triples.append(t)
        return triples

    def _generate_batch(self, texts: list[str]) -> list[str]:
        self._ensure_loaded()
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        output = self._model.generate(
            **inputs, num_beams=4, num_return_sequences=1, max_length=256
        )
        return self._tokenizer.batch_decode(output, skip_special_tokens=False)
