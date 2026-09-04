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


def smoke(adapter: Adapter, sentences: list[dict]) -> None:
    target = next((s for s in sentences if s.get("target")), sentences[0])
    print(f"[smoke] {adapter.name}: {target['text'][:70]}...")
    for triples in adapter.extract([target["text"]]):
        for t in dedupe(triples):
            print(f"  ({t.head!r}, {t.relation!r}, {t.tail!r}, {t.score:.3f})")
        if not triples:
            print("  (ไม่ได้ triple — stub หรือ model ว่าง)")


def run_full(adapter: Adapter, sentences: list[str]) -> None:
    triples_per_sent, ms = timed_extract(adapter, sentences)
    n = sum(len(dedupe(t)) for t in triples_per_sent)
    print(f"[run] {adapter.name}: {n} triples (dedupe แล้ว) · {ms:.1f} ms/ประโยค")


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
