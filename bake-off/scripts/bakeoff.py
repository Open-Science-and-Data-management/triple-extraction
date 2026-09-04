"""Bake-off r2: encoder zero-shot dedicated IE — one-off benchmark script.

ชุดประโยคจาก r1 (docs/bakeoff-results.md) ทุกประโยคมี category กำกับว่าวัดอะไร
(spec SPEC-bakeoff-r2.md) — report ต้องแตกผลตาม category ไม่รวมก้อนเดียว
"""

from __future__ import annotations

import json
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


def main() -> None:
    _, sentences = load_data()
    print(f"ประโยครวม {len(sentences)}")
    for cat in CATEGORIES:
        n = sum(1 for s in sentences if cat in s["categories"])
        status = "ok" if n >= 2 else "FAIL (<2)"
        print(f"  {cat:10s} {n:2d}  {status}")
    targets = [s["text"][:60] for s in sentences if s.get("target")]
    print(f"ประโยคเป้า: {targets}")


if __name__ == "__main__":
    main()
