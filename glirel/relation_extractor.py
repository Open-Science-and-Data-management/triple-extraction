"""Relation extraction: รับ paragraph (text) → คืน text เดิม + relations ที่ GLiREL เดาได้.

ใช้ spaCy สำหรับ NER + tokenization แล้วส่งต่อให้ GLiREL (spaCy pipeline component)
"""

import spacy
import glirel  # noqa: F401  (ลงทะเบียน spaCy factory "glirel")

# แก้ NER ของ spaCy ให้ถูก (เช่น {"SpaceX": "ORG"}) — component fix_ner อ่านจากตรงนี้ตอนรัน
NER_OVERRIDES: dict = {}

# relations ที่อยากให้โมเดลหา พร้อมข้อจำกัด head/tail entity type (dict ว่าง = ไม่จำกัด)
DEFAULT_LABELS = {
    "glirel_labels": {
        "no relation": {},
        "co-founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "parent": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "child": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "spouse": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "acquired by": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},
        "subsidiary of": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},
        "headquartered in": {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},
        "located in or next to body of water": {"allowed_head": ["LOC", "GPE", "FAC"], "allowed_tail": ["LOC", "GPE"]},
    }
}


@spacy.Language.component("fix_ner")
def _fix_ner(doc):
    """แก้ label ของ entity ตาม NER_OVERRIDES (รันระหว่าง ner กับ glirel).

    แก้ที่ token.ent_type_ โดยตรง — Span.label_ setter แก้ไม่ persist ลง doc.ents
    """
    for ent in doc.ents:
        label = NER_OVERRIDES.get(ent.text)
        if label:
            for token in ent:
                token.ent_type_ = label
    return doc


def _get_pipeline():
    """โหลด pipeline spaCy + component GLiREL ครั้งเดียว (ช้า ~1GB) แล้ว cache ไว้."""
    if not hasattr(_get_pipeline, "nlp"):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("fix_ner", after="ner")
        nlp.add_pipe("glirel", after="fix_ner")
        _get_pipeline.nlp = nlp
    return _get_pipeline.nlp


def extract_relations(text: str, labels: dict = DEFAULT_LABELS, threshold: float = 0.3) -> dict:
    """รับ paragraph เข้าไป คืน dict: {'text': <paragraph เดิม>, 'relations': [...]} เรียงตาม score.

    relations แต่ละตัว: {head_pos, tail_pos, head_text, tail_text, label, score}
    """
    doc = next(_get_pipeline().pipe([(text, labels)], as_tuples=True))[0]
    relations = sorted(doc._.relations, key=lambda r: r["score"], reverse=True)
    return {"text": text, "relations": relations}


if __name__ == "__main__":
    sample = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976. The company is headquartered in Cupertino, California."
    result = extract_relations(sample)
    print("TEXT:", result["text"])
    print("RELATIONS:")
    for r in result["relations"]:
        print(f"  {r['head_text']} --{r['label']}--> {r['tail_text']}  (score={r['score']:.3f})")
