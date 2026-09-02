"""ทดสอบ RebelExtractor: glue logic (stub โมเดล) + gpu smoke (โมเดลจริง)"""

import pytest

from triple_extraction.extractor.rebel import RebelExtractor

DOC = (
    "Marie Curie was born in Warsaw. "
    "She later moved to Paris, where she met Pierre Curie. "
    "The Eiffel Tower is located in Paris."
)


def make_stub_extractor(monkeypatch, decoded_output: str):
    """RebelExtractor ที่ใช้ spaCy จริง แต่ stub การเรียกโมเดล — รันได้ไม่ต้องมี GPU"""
    ex = RebelExtractor(batch_size=2)

    class StubModel:
        def generate(self, **kwargs):
            # 1 sequence ต่อ 1 input ใน batch
            n = kwargs["input_ids"].shape[0]
            return [[0] * n for _ in range(n)]

    ex._tokenizer = None
    ex._model = StubModel()
    ex._nlp = __import__("spacy").load("en_core_web_sm")

    def fake_generate_batch(texts):
        return [decoded_output] * len(texts)

    monkeypatch.setattr(ex, "_generate_batch", fake_generate_batch)
    return ex


def test_extract_attaches_sentence_index_and_offsets(monkeypatch):
    ex = make_stub_extractor(
        monkeypatch, "<triplet> Marie Curie <subj> Warsaw <obj> place of birth"
    )

    one_sentence = "Marie Curie was born in Warsaw."
    triples = ex.extract(one_sentence)

    assert triples == [
        {
            "head": "Marie Curie",
            "relation": "place of birth",
            "tail": "Warsaw",
            "sentence_index": 0,
            "start": 0,
            "end": len(one_sentence),
            "extractor": "rebel",
        }
    ]
    # offset ชี้เทียบตำแหน่งจริงใน input text
    assert one_sentence[triples[0]["start"] : triples[0]["end"]].startswith("Marie")


def test_extract_batches_all_sentences(monkeypatch):
    ex = make_stub_extractor(
        monkeypatch, "<triplet> Paris <subj> France <obj> capital of"
    )

    triples = ex.extract(DOC)  # 3 ประโยค, batch_size=2 → 2 รอบ batch

    assert len(triples) == 3
    assert [t["sentence_index"] for t in triples] == [0, 1, 2]


def test_device_defaults_to_cuda_when_available():
    torch = pytest.importorskip("torch")
    ex = RebelExtractor()

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert ex.device == expected


@pytest.mark.gpu
def test_real_model_extracts_triples_from_document():
    ex = RebelExtractor()

    triples = ex.extract(DOC)

    assert len(triples) > 0
    for t in triples:
        assert t["head"] and t["relation"] and t["tail"]
        assert t["extractor"] == "rebel"
        assert isinstance(t["sentence_index"], int)
