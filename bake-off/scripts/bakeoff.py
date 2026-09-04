"""Bake-off r2: encoder zero-shot dedicated IE — one-off benchmark script.

ชุดประโยคจาก r1 (docs/bakeoff-results.md) ทุกประโยคมี category กำกับว่าวัดอะไร
(spec SPEC-bakeoff-r2.md) — report ต้องแตกผลตาม category ไม่รวมก้อนเดียว
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# category ตาม spec: alias / comparison / training / benchmark / effect / multi-rel / hard
# ประโยคเป้าเกณฑ์ success (LoRA, reduces, hallucination) = alias+effect
# เติม 1 ประโยค (CoT/CoT alias) — r1 มีประโยค alias แค่ประโยคเดียว ขัดเงื่อนไข ≥2/category
SENTENCES: list[dict] = [
    {"text": "Low-Rank Adaptation, or LoRA, reduces hallucination in large language models by constraining weight updates to low-rank matrices.", "categories": ["alias", "effect"], "target": True},
    {"text": "Chain-of-thought prompting, also known as CoT, improves reasoning accuracy of large language models on multi-step problems.", "categories": ["alias", "effect"]},
    {"text": "We fine-tune LLaMA-7B on the Alpaca dataset using LoRA and evaluate on MMLU.", "categories": ["multi-rel"]},
    {"text": "GPT-4 outperforms GPT-3.5 on most benchmarks, achieving 86.4% on MMLU.", "categories": ["comparison", "benchmark"]},
    {"text": "The attention mechanism introduced by Vaswani et al. replaces recurrence entirely with self-attention.", "categories": ["hard"]},
    {"text": "BERT was pretrained on the BookCorpus and English Wikipedia using masked language modeling.", "categories": ["training"]},
    {"text": "RLHF aligns language model behavior with human preferences by optimizing against a learned reward model.", "categories": ["effect"]},
    {"text": "InstructGPT demonstrates that reinforcement learning from human feedback improves truthfulness of GPT-3 outputs.", "categories": ["effect"]},
    {"text": "Chain-of-thought prompting enables large language models to solve multi-step arithmetic reasoning problems.", "categories": ["effect"]},
    {"text": "We achieve 94.8 F1 on SQuAD 2.0 using RoBERTa with a span prediction head.", "categories": ["benchmark"]},
    {"text": "FlashAttention reduces memory usage of attention computation from quadratic to linear in sequence length.", "categories": ["effect"]},
    {"text": "Llama 2 introduces a family of models trained on 2 trillion tokens with grouped-query attention.", "categories": ["multi-rel"]},
    {"text": "Retrieval-augmented generation grounds model answers in documents retrieved by DPR from Wikipedia.", "categories": ["hard"]},
    {"text": "Mistral-7B uses sliding window attention and outperforms Llama 2 13B on many tasks.", "categories": ["comparison"]},
    {"text": "Qwen2.5 was evaluated on GSM8K, HumanEval, and MMLU, surpassing Qwen2 across all metrics.", "categories": ["comparison"]},
    {"text": "Speculative decoding accelerates inference by drafting tokens with a smaller model and verifying them with the target model.", "categories": ["multi-rel"]},
    {"text": "The T5 model treats all NLP tasks as text-to-text problems using the span corruption objective.", "categories": ["hard"]},
    {"text": "DeepSeek-R1 distills reasoning ability from large reinforcement-learned models into dense models of 1.5B to 70B parameters.", "categories": ["hard"]},
    {"text": "We report that LoRA matches full fine-tuning performance on GLUE while training 10,000 times fewer parameters.", "categories": ["hard"]},
    {"text": "LoRA fine-tunes only a small subset of parameters by adding trainable rank decomposition matrices to each transformer layer.", "categories": ["training"]},
    {"text": "Quantization to 4-bit precision reduces GPU memory requirements of large language models with minimal accuracy loss.", "categories": ["effect"]},
    {"text": "Direct preference optimization fine-tunes the policy directly on preference data without an explicit reward model.", "categories": ["training"]},
    {"text": "Scaling laws show that loss decreases predictably as model size, dataset size, and compute increase together.", "categories": ["hard"]},
    {"text": "DeBERTa improves upon RoBERTa using disentangled attention and achieves 88.8% accuracy on MNLI.", "categories": ["comparison", "benchmark"]},
    {"text": "BART combines bidirectional encoding with autoregressive generation, pretrained on text infilling objectives.", "categories": ["training"]},
    {"text": "PaLM was trained on 780 billion tokens of high-quality text using the Pathways system across 6144 TPU v4 chips.", "categories": ["training"]},
    {"text": "Knowledge distillation transfers the behavior of a large teacher model into a smaller student model such as DistilBERT.", "categories": ["hard"]},
    {"text": "Self-consistency decodes multiple reasoning paths and takes a majority vote, raising GSM8K accuracy of PaLM-540B from 56% to 74%.", "categories": ["multi-rel"]},
]

CATEGORIES = ["alias", "comparison", "training", "benchmark", "effect", "multi-rel", "hard"]


def load_schema() -> dict:
    """โหลด seed schema (relation hints 10 + entity labels) จาก schema/seed.json"""
    return json.loads((ROOT / "schema" / "seed.json").read_text())


def load_data() -> tuple[dict, list[dict]]:
    """คืน (schema, sentences) — ตรวจความครบถ้วนของข้อมูลตั้งแต่โหลด"""
    schema = load_schema()
    assert len(schema["relation_hints"]) == 10, f"relation hints ต้องมี 10 ได้ {len(schema['relation_hints'])}"
    assert schema["entity_labels"], "entity labels ห้ามว่าง"
    for s in SENTENCES:
        assert s["categories"], f"ประโยคไม่มี category: {s['text'][:50]}"
        unknown = set(s["categories"]) - set(CATEGORIES)
        assert not unknown, f"category ไม่รู้จัก {unknown}: {s['text'][:50]}"
    return schema, SENTENCES


MODELS = ["gliner-relex", "glirel", "gliner-pyrheads", "relik", "nuextract"]


@dataclass
class Triple:
    """score บังคับไม่มี default — แกนของ rate-once-slice-many"""

    head: str
    relation: str
    tail: str
    score: float


@dataclass
class Adapter:
    name: str
    extract: Callable[[list[str]], list[list[Triple]]]


# stub คืน [] ทุกประโยค — ทยอยแทนด้วย loader จริงใน Task 4–7
# ponytail: dict ชื่อ → factory เพราะ model โหลด lazy ตอนถูกเลือกเท่านั้น
def _stub_extract(sentences: list[str]) -> list[list[Triple]]:
    return [[] for _ in sentences]


ADAPTER_FACTORIES: dict[str, Callable[[], Adapter]] = {m: (lambda m=m: Adapter(m, _stub_extract)) for m in MODELS}


def make_gliner_relex() -> Adapter:
    from gliner import GLiNER

    schema = load_schema()
    model = GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0").to("cuda").eval()
    labels, relations = schema["entity_labels"], schema["relation_hints"]

    def extract(sentences: list[str]) -> list[list[Triple]]:
        # threshold ต่ำ ๆ ตาม r1 "lo" เพื่อเก็บ raw scores ไว้ slice ทีหลัง (rate-once-slice-many)
        _, rels = model.inference(
            sentences, labels, relations=relations, threshold=0.3, adjacency_threshold=0.3,
            relation_threshold=0.3, batch_size=8, return_relations=True,
        )
        return [[Triple(r["head"]["text"], r["relation"], r["tail"]["text"], float(r["score"])) for r in rs] for rs in rels]

    return Adapter("gliner-relex", extract)


ADAPTER_FACTORIES["gliner-relex"] = make_gliner_relex


def make_nuextract() -> Adapter:
    import torch
    from transformers import AutoModelForVision2Seq, AutoTokenizer

    ckpt = "numind/NuExtract-2.0-2B"  # spec ระบุ 1.5B ซึ่งไม่มีบน HF — human อนุมัติใช้ 2B fp16
    # ไม่ใช้ device_map (กันเพิ่ม accelerate) — โหลดแล้วค่อยย้ายขึ้น GPU
    model = AutoModelForVision2Seq.from_pretrained(ckpt, torch_dtype=torch.float16).eval().to("cuda")
    tok = AutoTokenizer.from_pretrained(ckpt)
    schema = load_schema()
    # "verbatim-string" ไม่งั้น model จะ nest tail เป็น object ตามราย type ในคำอธิบาย
    template = json.dumps({"triples": [{
        "head": "verbatim-string (short entity mention)",
        "relation": "one of: " + " | ".join(schema["relation_hints"]),
        "tail": "verbatim-string (short entity mention)",
    }]})

    def _token_char_probs(out_ids, scores) -> tuple[str, list[tuple[int, int, float]]]:
        """map token → ช่วง char ใน output text + prob ของ token (greedy)"""
        text = tok.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tok_probs = torch.softmax(torch.stack(scores, 0), -1).max(-1).values
        ranges, prev = [], 0
        for i in range(len(out_ids)):
            end = len(tok.decode(out_ids[:i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False))
            ranges.append((prev, end, float(tok_probs[i])))  # ponytail: decode ซ้ำ O(n²) — output สั้นจึงช่างมัน
            prev = end
        return text, ranges

    def _object_spans(text: str) -> list[tuple[int, int]]:
        """ช่วง char ของ {...} ทุกชั้นใน output (brace matching ไม่ง้อ key order)"""
        spans, stack = [], []
        for i, ch in enumerate(text):
            if ch == "{":
                stack.append(i)
            elif ch == "}" and stack:
                spans.append((stack.pop(), i + 1))
        return spans

    def parse(out_ids, scores) -> list[Triple]:
        text, ranges = _token_char_probs(out_ids, scores)
        out = []
        for a, b in _object_spans(text):
            try:
                obj = json.loads(text[a:b])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and {"head", "relation", "tail"} <= obj.keys():
                ps = [p for s, e, p in ranges if e > a and s < b]  # token ที่ overlap ช่วง object
                score = sum(ps) / len(ps) if ps else 0.0
                out.append(Triple(str(obj["head"]), str(obj["relation"]), str(obj["tail"]), score))
        return out

    def extract(sentences: list[str]) -> list[list[Triple]]:
        results = []
        for s in sentences:  # ponytail: batch=1 ต่อประโยค ตาม pipeline จริง (spaCy แบ่งแล้วยิงทีละประโยค)
            with torch.inference_mode():
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": s}], template=template, tokenize=False, add_generation_prompt=True)
                inputs = tok([prompt], return_tensors="pt").to("cuda")
                gen = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1024,
                                     output_scores=True, return_dict_in_generate=True)
                results.append(parse(gen.sequences[0][inputs["input_ids"].shape[1]:], gen.scores))
        return results

    return Adapter("nuextract", extract)


ADAPTER_FACTORIES["nuextract"] = make_nuextract


def make_gliner_pyrheads() -> Adapter:
    from gliner2 import GLiNER2

    # checkpoint ใน spec (gliner-pyrheads-large-v0.5) ไม่มีบน HF แล้ว —
    # human อนุมัติ gliner2 + fastino/gliner2-large-v1
    model = GLiNER2.from_pretrained("fastino/gliner2-large-v1").to("cuda")
    schema = load_schema()
    entities, relations = schema["entity_labels"], schema["relation_hints"]
    gliner2_schema = model.create_schema().entities(entities).relations(relations, threshold=0.3)

    def extract(sentences: list[str]) -> list[list[Triple]]:
        out = model.batch_extract(sentences, gliner2_schema, threshold=0.3, include_confidence=True)
        results = []
        for o in out:
            ts = []
            for rel, heads_tails in o.get("relation_extraction", {}).items():
                for ht in heads_tails:
                    # gliner2 ไม่มี score ระดับ triple — ใช้ค่าเฉลี่ย confidence ของ head/tail
                    score = (ht["head"]["confidence"] + ht["tail"]["confidence"]) / 2
                    ts.append(Triple(ht["head"]["text"], rel, ht["tail"]["text"], score))
            results.append(ts)
        return results

    return Adapter("gliner-pyrheads", extract)


ADAPTER_FACTORIES["gliner-pyrheads"] = make_gliner_pyrheads


def make_glirel() -> Adapter:
    import re

    from gliner import GLiNER
    from glirel import GLiREL

    schema = load_schema()
    # NER จาก GLiNER checkpoint สายเดียวกับ Relex (ล็อกตั้งแต่ config) — GLiREL ไม่หา entity เอง
    ner_model = GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0").to("cuda").eval()
    model = GLiREL.from_pretrained("jackboyla/glirel-large-v0").to("cuda").eval()
    labels, relations = schema["entity_labels"], schema["relation_hints"]

    def _tokenize(text: str) -> list[str]:
        # tokenizer เดียวกับ batch_predict_relations ของ glirel เพื่อ map ตำแหน่งให้ตรง
        return [m.group() for m in re.finditer(r"\w+(?:[-_]\w+)*|\S", text)]

    def _ner_token_spans(text: str) -> list[list]:
        # NER char span → token span [start, end(inclusive), label, text]
        ents = ner_model.inference([text], labels, threshold=0.3)[0]
        # gliner คืน shape ไม่คงที่ — บน cuda มีการ wrap list ซ้อนอีกชั้น
        while ents and isinstance(ents[0], list):
            ents = ents[0]
        toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+(?:[-_]\w+)*|\S", text)]
        out = []
        for e in ents:
            idx = [i for i, (_, s, t) in enumerate(toks) if s < e["end"] and e["start"] < t]
            if idx:
                out.append([idx[0], idx[-1], e["label"], text[e["start"]:e["end"]]])
        return out

    def extract(sentences: list[str]) -> list[list[Triple]]:
        ners = [_ner_token_spans(s) for s in sentences]
        # threshold ต่ำเก็บ raw scores ไว้ slice ทีหลัง (rate-once-slice-many)
        rels = model.batch_predict_relations(sentences, relations, threshold=0.3, ner=ners, top_k=-1)
        out = []
        for sent, rs in zip(sentences, rels):
            toks = _tokenize(sent)
            ts = []
            for r in rs:
                # glirel บวก +1 ท้าย position เพื่อคุยกับ spaCy tokenization — ตัดกลับ
                head = " ".join(toks[r["head_pos"][0] - 1:r["head_pos"][1]])
                tail = " ".join(toks[r["tail_pos"][0] - 1:r["tail_pos"][1]])
                if head and tail:  # span ชนขอบประโยคได้ entity ว่าง — ทิ้งเป็น noise
                    ts.append(Triple(head, r["label"], tail, float(r["score"])))
            out.append(ts)
        return out

    return Adapter("glirel", extract)


ADAPTER_FACTORIES["glirel"] = make_glirel


# ผล smoke (Task 6): ให้ seed relation เป็น candidates แล้ว reader ทำนาย 0 triple ทุกกรณี
# (รวมลอง format <def>) — ReLiK ปิด schema ยืนยัน รันด้วย native NYT relations แทน = ข้อมูลของแผนที่
RELIK_CLOSED_SCHEMA = True


def make_relik() -> Adapter:
    from relik import Relik

    # transformers 4.52 init ทุก model บน meta device เสมอ (get_init_context) →
    # relik resize embeddings ใน __init__ ก่อน weights โหลด ทำให้ embedding ว่างเป็นศูนย์
    # → ปิด meta init ให้ init จริงบน CPU (RAM ~2GB ชั่วคราว ไม่กระทบ VRAM)
    from transformers.modeling_utils import PreTrainedModel, no_init_weights

    PreTrainedModel.get_init_context = classmethod(
        lambda cls, *a, **kw: [no_init_weights()])

    relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large", device="cuda")

    def extract(sentences: list[str]) -> list[list[Triple]]:
        results = []
        for s in sentences:  # ponytail: batch list ชนบั๊ก candidates ของ relik (w._d) — ยิงทีละประโยค
            # window_size="none": ประโยคสั้นไม่ต้อง window + กลบบั๊ก merge_windows ของ SpacySentenceSplitter
            out = relik(s, top_k=24, window_size="none", progress_bar=False)
            results.append([Triple(t.subject.text, t.label, t.object.text, float(t.confidence)) for t in out.triples])
        return results

    return Adapter("relik", extract)


ADAPTER_FACTORIES["relik"] = make_relik


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("ต้องรันบน GPU เท่านั้น — torch.cuda.is_available() == False")


def dedupe(triples: list[Triple]) -> list[Triple]:
    """ตัด triple ซ้ำเป๊ะในประโยคเดียวกัน เก็บตัว score สูงสุด"""
    best: dict[tuple[str, str, str], Triple] = {}
    for t in triples:
        key = (t.head, t.relation, t.tail)
        if key not in best or t.score > best[key].score:
            best[key] = t
    return list(best.values())


def timed_extract(adapter: Adapter, sentences: list[str]) -> tuple[list[list[Triple]], float]:
    """warm-up 1 ประโยคก่อน แล้วจับเวลารวม → ms/ประโยค"""
    adapter.extract(sentences[:1])
    t0 = time.perf_counter()
    out = adapter.extract(sentences)
    return out, (time.perf_counter() - t0) * 1000 / len(sentences)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bake-off r2: encoder zero-shot dedicated IE")
    p.add_argument("--smoke", choices=MODELS, metavar="MODEL", help="smoke 1 ประโยคเป้า ของ model เดียว")
    p.add_argument("--only", choices=MODELS, metavar="MODEL", help="รันเต็มเฉพาะ model เดียว (default: ทุก model)")
    return p.parse_args()


def vram_peak() -> str:
    import torch

    return f"VRAM peak: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"


def smoke(adapter: Adapter, sentences: list[dict]) -> None:
    target = next((s for s in sentences if s.get("target")), sentences[0])
    print(f"[smoke] {adapter.name}: {target['text'][:70]}...")
    for triples in adapter.extract([target["text"]]):
        for t in dedupe(triples):
            print(f"  ({t.head!r}, {t.relation!r}, {t.tail!r}, {t.score:.3f})")
        if not triples:
            print("  (ไม่ได้ triple — stub หรือ model ว่าง)")
    print(f"  {vram_peak()}")


def run_full(adapter: Adapter, sentences: list[str]) -> None:
    triples_per_sent, ms = timed_extract(adapter, sentences)
    n = sum(len(dedupe(t)) for t in triples_per_sent)
    print(f"[run] {adapter.name}: {n} triples (dedupe แล้ว) · {ms:.1f} ms/ประโยค · {vram_peak()}")


def main() -> None:
    args = parse_args()
    _, sentences = load_data()
    require_cuda()
    if args.smoke:
        smoke(ADAPTER_FACTORIES[args.smoke](), sentences)
        return
    names = [args.only] if args.only else MODELS
    texts = [s["text"] for s in sentences]
    for name in names:
        run_full(ADAPTER_FACTORIES[name](), texts)


if __name__ == "__main__":
    main()
