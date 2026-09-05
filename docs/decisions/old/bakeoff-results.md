# Bake-off: GLiNER-Relex vs NuExtract (seed schema ชุดเดียวกัน)

relations (hint): ['evaluated on', 'outperforms', 'uses dataset', 'achieves metric', 'uses method', 'introduces', 'pretrained on', 'fine-tuned on', 'reduces', 'improves']

## gliner-relex lo 0.3/0.5/0.5 (device=cuda)

**1.** Low-Rank Adaptation, or LoRA, reduces hallucination in large language models by constraining weight updates to low-rank matrices.

- (Low-Rank Adaptation) —[uses method]→ (LoRA)
- (Low-Rank Adaptation) —[reduces]→ (hallucination)
- (Low-Rank Adaptation) —[improves]→ (hallucination)
- (LoRA) —[reduces]→ (hallucination)
- (weight updates) —[pretrained on]→ (large language models)
- (weight updates) —[pretrained on]→ (low-rank matrices)
- (weight updates) —[fine-tuned on]→ (low-rank matrices)
- _75 ms, 7 triples_

**2.** We fine-tune LLaMA-7B on the Alpaca dataset using LoRA and evaluate on MMLU.

- (LLaMA-7B) —[pretrained on]→ (Alpaca)
- (LLaMA-7B) —[fine-tuned on]→ (Alpaca)
- (LLaMA-7B) —[uses method]→ (LoRA)
- (LLaMA-7B) —[evaluated on]→ (MMLU)
- _18 ms, 4 triples_

**3.** GPT-4 outperforms GPT-3.5 on most benchmarks, achieving 86.4% on MMLU.

- (GPT-4) —[outperforms]→ (GPT-3.5)
- (GPT-4) —[achieves metric]→ (86.4%)
- (GPT-4) —[evaluated on]→ (MMLU)
- (GPT-3.5) —[achieves metric]→ (86.4%)
- (GPT-3.5) —[evaluated on]→ (MMLU)
- (86.4%) —[evaluated on]→ (MMLU)
- _72 ms, 6 triples_

**4.** The attention mechanism introduced by Vaswani et al. replaces recurrence entirely with self-attention.

- (attention mechanism) —[reduces]→ (recurrence)
- (attention mechanism) —[improves]→ (recurrence)
- (attention mechanism) —[reduces]→ (self-attention)
- _155 ms, 3 triples_

**5.** BERT was pretrained on the BookCorpus and English Wikipedia using masked language modeling.

- (BERT) —[pretrained on]→ (BookCorpus)
- (BERT) —[fine-tuned on]→ (BookCorpus)
- (BERT) —[pretrained on]→ (English Wikipedia)
- (BERT) —[fine-tuned on]→ (English Wikipedia)
- (BERT) —[uses method]→ (masked language modeling)
- _18 ms, 5 triples_

**6.** RLHF aligns language model behavior with human preferences by optimizing against a learned reward model.

- (RLHF) —[uses method]→ (RLHF)
- (RLHF) —[improves]→ (language model behavior)
- (RLHF) —[achieves metric]→ (human preferences)
- (RLHF) —[improves]→ (human preferences)
- _18 ms, 4 triples_

**7.** InstructGPT demonstrates that reinforcement learning from human feedback improves truthfulness of GPT-3 outputs.

- (reinforcement learning) —[improves]→ (truthfulness)
- (reinforcement learning) —[improves]→ (GPT-3 outputs)
- (human feedback) —[improves]→ (truthfulness)
- _18 ms, 3 triples_

**8.** Chain-of-thought prompting enables large language models to solve multi-step arithmetic reasoning problems.

- (Chain-of-thought prompting) —[improves]→ (multi-step arithmetic reasoning problems)
- (large language models) —[achieves metric]→ (multi-step arithmetic reasoning problems)
- _18 ms, 2 triples_

**9.** We achieve 94.8 F1 on SQuAD 2.0 using RoBERTa with a span prediction head.

- (94.8 F1) —[evaluated on]→ (SQuAD 2.0)
- _17 ms, 1 triples_

**10.** FlashAttention reduces memory usage of attention computation from quadratic to linear in sequence length.

- (FlashAttention) —[reduces]→ (memory usage)
- (FlashAttention) —[uses method]→ (attention computation)
- (FlashAttention) —[uses method]→ (quadratic)
- (FlashAttention) —[achieves metric]→ (sequence length)
- (FlashAttention) —[improves]→ (sequence length)
- (attention computation) —[achieves metric]→ (sequence length)
- _17 ms, 6 triples_

**11.** Llama 2 introduces a family of models trained on 2 trillion tokens with grouped-query attention.

- (Llama 2) —[introduces]→ (family of models)
- (Llama 2) —[uses method]→ (grouped-query attention)
- (Llama 2) —[improves]→ (grouped-query attention)
- (family of models) —[pretrained on]→ (2 trillion tokens)
- (family of models) —[fine-tuned on]→ (2 trillion tokens)
- (family of models) —[achieves metric]→ (grouped-query attention)
- (family of models) —[uses method]→ (grouped-query attention)
- _18 ms, 7 triples_

**12.** Retrieval-augmented generation grounds model answers in documents retrieved by DPR from Wikipedia.

- (Retrieval-augmented generation) —[introduces]→ (grounds model)
- (Retrieval-augmented generation) —[improves]→ (grounds model)
- (Retrieval-augmented generation) —[reduces]→ (answers)
- (Retrieval-augmented generation) —[improves]→ (answers)
- (Retrieval-augmented generation) —[fine-tuned on]→ (documents)
- _17 ms, 5 triples_

**13.** Mistral-7B uses sliding window attention and outperforms Llama 2 13B on many tasks.

- (Mistral-7B) —[uses method]→ (sliding window attention)
- (Mistral-7B) —[outperforms]→ (Llama 2 13B)
- (Mistral-7B) —[evaluated on]→ (many tasks)
- (Mistral-7B) —[fine-tuned on]→ (many tasks)
- (Llama 2 13B) —[evaluated on]→ (many tasks)
- _17 ms, 5 triples_

**14.** Qwen2.5 was evaluated on GSM8K, HumanEval, and MMLU, surpassing Qwen2 across all metrics.

- (Qwen2.5) —[evaluated on]→ (GSM8K)
- (Qwen2.5) —[evaluated on]→ (HumanEval)
- (Qwen2.5) —[evaluated on]→ (MMLU)
- (Qwen2.5) —[outperforms]→ (Qwen2)
- (Qwen2.5) —[achieves metric]→ (Qwen2)
- _18 ms, 5 triples_

**15.** Speculative decoding accelerates inference by drafting tokens with a smaller model and verifying them with the target model.

- (Speculative decoding) —[reduces]→ (inference)
- (Speculative decoding) —[improves]→ (inference)
- (Speculative decoding) —[improves]→ (tokens)
- _19 ms, 3 triples_

**16.** The T5 model treats all NLP tasks as text-to-text problems using the span corruption objective.

- (T5 model) —[uses method]→ (NLP)
- (T5 model) —[achieves metric]→ (span corruption objective)
- (T5 model) —[improves]→ (span corruption objective)
- _18 ms, 3 triples_

**17.** DeepSeek-R1 distills reasoning ability from large reinforcement-learned models into dense models of 1.5B to 70B parameters.

- (DeepSeek-R1) —[reduces]→ (reasoning ability)
- (DeepSeek-R1) —[improves]→ (reasoning ability)
- (dense models) —[achieves metric]→ (70B)
- _18 ms, 3 triples_

**18.** We report that LoRA matches full fine-tuning performance on GLUE while training 10,000 times fewer parameters.

- (LoRA) —[uses method]→ (full fine-tuning)
- (LoRA) —[achieves metric]→ (performance)
- (LoRA) —[improves]→ (performance)
- (full fine-tuning) —[fine-tuned on]→ (GLUE)
- (performance) —[evaluated on]→ (GLUE)
- (performance) —[fine-tuned on]→ (GLUE)
- _18 ms, 6 triples_

**19.** LoRA fine-tunes only a small subset of parameters by adding trainable rank decomposition matrices to each transformer layer.

- (LoRA) —[uses method]→ (LoRA)
- (LoRA) —[reduces]→ (parameters)
- (LoRA) —[improves]→ (parameters)
- (LoRA) —[introduces]→ (rank decomposition matrices)
- (LoRA) —[uses dataset]→ (transformer layer)
- (rank decomposition matrices) —[pretrained on]→ (transformer layer)
- (rank decomposition matrices) —[fine-tuned on]→ (transformer layer)
- _19 ms, 7 triples_

**20.** Quantization to 4-bit precision reduces GPU memory requirements of large language models with minimal accuracy loss.

- (Quantization) —[reduces]→ (GPU memory requirements)
- (Quantization) —[improves]→ (GPU memory requirements)
- _19 ms, 2 triples_

**21.** Direct preference optimization fine-tunes the policy directly on preference data without an explicit reward model.

- (Direct preference optimization) —[reduces]→ (policy)
- (Direct preference optimization) —[improves]→ (policy)
- (Direct preference optimization) —[fine-tuned on]→ (preference data)
- (policy) —[evaluated on]→ (preference data)
- (policy) —[fine-tuned on]→ (preference data)
- _17 ms, 5 triples_

**22.** Scaling laws show that loss decreases predictably as model size, dataset size, and compute increase together.

- (Scaling laws) —[introduces]→ (loss)
- (Scaling laws) —[reduces]→ (loss)
- (model) —[achieves metric]→ (size)
- (model) —[uses dataset]→ (dataset)
- _18 ms, 4 triples_

**23.** DeBERTa improves upon RoBERTa using disentangled attention and achieves 88.8% accuracy on MNLI.

- (DeBERTa) —[outperforms]→ (RoBERTa)
- (DeBERTa) —[improves]→ (RoBERTa)
- (DeBERTa) —[uses method]→ (disentangled attention)
- (DeBERTa) —[achieves metric]→ (88.8% accuracy)
- (DeBERTa) —[evaluated on]→ (MNLI)
- (RoBERTa) —[uses method]→ (disentangled attention)
- (RoBERTa) —[achieves metric]→ (88.8% accuracy)
- (88.8% accuracy) —[evaluated on]→ (MNLI)
- _17 ms, 8 triples_

**24.** BART combines bidirectional encoding with autoregressive generation, pretrained on text infilling objectives.

- (BART) —[uses method]→ (bidirectional encoding)
- (BART) —[introduces]→ (bidirectional encoding)
- (BART) —[uses method]→ (autoregressive generation)
- (BART) —[introduces]→ (autoregressive generation)
- (BART) —[evaluated on]→ (text infilling objectives)
- (BART) —[pretrained on]→ (text infilling objectives)
- (BART) —[fine-tuned on]→ (text infilling objectives)
- (autoregressive generation) —[pretrained on]→ (text infilling objectives)
- (autoregressive generation) —[fine-tuned on]→ (text infilling objectives)
- _19 ms, 9 triples_

**25.** PaLM was trained on 780 billion tokens of high-quality text using the Pathways system across 6144 TPU v4 chips.

- (PaLM) —[pretrained on]→ (780 billion tokens)
- (PaLM) —[pretrained on]→ (high-quality text)
- (PaLM) —[uses method]→ (Pathways system)
- (PaLM) —[uses dataset]→ (6144 TPU v4 chips)
- _19 ms, 4 triples_

**26.** Knowledge distillation transfers the behavior of a large teacher model into a smaller student model such as DistilBERT.

- (Knowledge distillation) —[reduces]→ (behavior)
- (Knowledge distillation) —[improves]→ (behavior)
- (Knowledge distillation) —[improves]→ (large teacher model)
- (Knowledge distillation) —[improves]→ (smaller student model)
- (smaller student model) —[outperforms]→ (DistilBERT)
- _18 ms, 5 triples_

**27.** Self-consistency decodes multiple reasoning paths and takes a majority vote, raising GSM8K accuracy of PaLM-540B from 56% to 74%.

- (Self-consistency) —[introduces]→ (Self-consistency)
- (Self-consistency) —[uses method]→ (multiple reasoning paths)
- (Self-consistency) —[uses method]→ (majority vote)
- (Self-consistency) —[improves]→ (majority vote)
- (GSM8K) —[achieves metric]→ (accuracy)
- (PaLM-540B) —[achieves metric]→ (accuracy)
- (PaLM-540B) —[achieves metric]→ (56%)
- (PaLM-540B) —[achieves metric]→ (74%)
- _21 ms, 8 triples_

## gliner-relex mid 0.5/0.6/0.7 (device=cuda)

**1.** Low-Rank Adaptation, or LoRA, reduces hallucination in large language models by constraining weight updates to low-rank matrices.

- (Low-Rank Adaptation) —[uses method]→ (LoRA)
- (Low-Rank Adaptation) —[reduces]→ (hallucination)
- (weight updates) —[pretrained on]→ (low-rank matrices)
- (weight updates) —[fine-tuned on]→ (low-rank matrices)
- _18 ms, 4 triples_

**2.** We fine-tune LLaMA-7B on the Alpaca dataset using LoRA and evaluate on MMLU.

- (LLaMA-7B) —[pretrained on]→ (Alpaca)
- (LLaMA-7B) —[fine-tuned on]→ (Alpaca)
- (LLaMA-7B) —[evaluated on]→ (MMLU)
- _17 ms, 3 triples_

**3.** GPT-4 outperforms GPT-3.5 on most benchmarks, achieving 86.4% on MMLU.

- (GPT-4) —[outperforms]→ (GPT-3.5)
- (GPT-4) —[achieves metric]→ (86.4%)
- (GPT-4) —[evaluated on]→ (MMLU)
- (GPT-3.5) —[achieves metric]→ (86.4%)
- (GPT-3.5) —[evaluated on]→ (MMLU)
- _17 ms, 5 triples_

**4.** The attention mechanism introduced by Vaswani et al. replaces recurrence entirely with self-attention.

- (attention mechanism) —[reduces]→ (recurrence)
- _17 ms, 1 triples_

**5.** BERT was pretrained on the BookCorpus and English Wikipedia using masked language modeling.

- (BERT) —[pretrained on]→ (BookCorpus)
- (BERT) —[fine-tuned on]→ (BookCorpus)
- (BERT) —[pretrained on]→ (English Wikipedia)
- (BERT) —[fine-tuned on]→ (English Wikipedia)
- (BERT) —[uses method]→ (masked language modeling)
- _16 ms, 5 triples_

**6.** RLHF aligns language model behavior with human preferences by optimizing against a learned reward model.

- (RLHF) —[improves]→ (language model behavior)
- (RLHF) —[improves]→ (human preferences)
- _16 ms, 2 triples_

**7.** InstructGPT demonstrates that reinforcement learning from human feedback improves truthfulness of GPT-3 outputs.

- (reinforcement learning) —[improves]→ (truthfulness)
- _16 ms, 1 triples_

**8.** Chain-of-thought prompting enables large language models to solve multi-step arithmetic reasoning problems.

- (ไม่ได้ triple)
- _16 ms, 0 triples_

**9.** We achieve 94.8 F1 on SQuAD 2.0 using RoBERTa with a span prediction head.

- (94.8 F1) —[evaluated on]→ (SQuAD 2.0)
- _16 ms, 1 triples_

**10.** FlashAttention reduces memory usage of attention computation from quadratic to linear in sequence length.

- (FlashAttention) —[reduces]→ (memory usage)
- (FlashAttention) —[uses method]→ (attention computation)
- (FlashAttention) —[achieves metric]→ (sequence length)
- _15 ms, 3 triples_

**11.** Llama 2 introduces a family of models trained on 2 trillion tokens with grouped-query attention.

- (Llama 2) —[introduces]→ (family of models)
- (family of models) —[pretrained on]→ (2 trillion tokens)
- (family of models) —[fine-tuned on]→ (2 trillion tokens)
- _16 ms, 3 triples_

**12.** Retrieval-augmented generation grounds model answers in documents retrieved by DPR from Wikipedia.

- (Retrieval-augmented generation) —[improves]→ (answers)
- (Retrieval-augmented generation) —[fine-tuned on]→ (documents)
- _16 ms, 2 triples_

**13.** Mistral-7B uses sliding window attention and outperforms Llama 2 13B on many tasks.

- (Mistral-7B) —[uses method]→ (sliding window attention)
- (Mistral-7B) —[outperforms]→ (Llama 2 13B)
- _18 ms, 2 triples_

**14.** Qwen2.5 was evaluated on GSM8K, HumanEval, and MMLU, surpassing Qwen2 across all metrics.

- (Qwen2.5) —[evaluated on]→ (GSM8K)
- (Qwen2.5) —[evaluated on]→ (HumanEval)
- (Qwen2.5) —[evaluated on]→ (MMLU)
- (Qwen2.5) —[outperforms]→ (Qwen2)
- _17 ms, 4 triples_

**15.** Speculative decoding accelerates inference by drafting tokens with a smaller model and verifying them with the target model.

- (Speculative decoding) —[improves]→ (inference)
- _17 ms, 1 triples_

**16.** The T5 model treats all NLP tasks as text-to-text problems using the span corruption objective.

- (T5 model) —[uses method]→ (NLP)
- (T5 model) —[achieves metric]→ (span corruption objective)
- _16 ms, 2 triples_

**17.** DeepSeek-R1 distills reasoning ability from large reinforcement-learned models into dense models of 1.5B to 70B parameters.

- (DeepSeek-R1) —[reduces]→ (reasoning ability)
- (DeepSeek-R1) —[improves]→ (reasoning ability)
- (dense models) —[achieves metric]→ (70B)
- _17 ms, 3 triples_

**18.** We report that LoRA matches full fine-tuning performance on GLUE while training 10,000 times fewer parameters.

- (LoRA) —[uses method]→ (full fine-tuning)
- (LoRA) —[achieves metric]→ (performance)
- (LoRA) —[improves]→ (performance)
- _17 ms, 3 triples_

**19.** LoRA fine-tunes only a small subset of parameters by adding trainable rank decomposition matrices to each transformer layer.

- (LoRA) —[reduces]→ (parameters)
- (LoRA) —[improves]→ (parameters)
- _17 ms, 2 triples_

**20.** Quantization to 4-bit precision reduces GPU memory requirements of large language models with minimal accuracy loss.

- (Quantization) —[reduces]→ (GPU memory requirements)
- (Quantization) —[improves]→ (GPU memory requirements)
- _17 ms, 2 triples_

**21.** Direct preference optimization fine-tunes the policy directly on preference data without an explicit reward model.

- (Direct preference optimization) —[reduces]→ (policy)
- (Direct preference optimization) —[improves]→ (policy)
- (policy) —[evaluated on]→ (preference data)
- (policy) —[fine-tuned on]→ (preference data)
- _16 ms, 4 triples_

**22.** Scaling laws show that loss decreases predictably as model size, dataset size, and compute increase together.

- (Scaling laws) —[reduces]→ (loss)
- (model) —[uses dataset]→ (dataset)
- _16 ms, 2 triples_

**23.** DeBERTa improves upon RoBERTa using disentangled attention and achieves 88.8% accuracy on MNLI.

- (DeBERTa) —[outperforms]→ (RoBERTa)
- (DeBERTa) —[improves]→ (RoBERTa)
- (DeBERTa) —[uses method]→ (disentangled attention)
- (DeBERTa) —[achieves metric]→ (88.8% accuracy)
- (RoBERTa) —[uses method]→ (disentangled attention)
- _16 ms, 5 triples_

**24.** BART combines bidirectional encoding with autoregressive generation, pretrained on text infilling objectives.

- (BART) —[uses method]→ (bidirectional encoding)
- (BART) —[introduces]→ (bidirectional encoding)
- (BART) —[uses method]→ (autoregressive generation)
- (BART) —[introduces]→ (autoregressive generation)
- _16 ms, 4 triples_

**25.** PaLM was trained on 780 billion tokens of high-quality text using the Pathways system across 6144 TPU v4 chips.

- (PaLM) —[pretrained on]→ (780 billion tokens)
- (PaLM) —[uses method]→ (Pathways system)
- _17 ms, 2 triples_

**26.** Knowledge distillation transfers the behavior of a large teacher model into a smaller student model such as DistilBERT.

- (Knowledge distillation) —[reduces]→ (behavior)
- (Knowledge distillation) —[improves]→ (behavior)
- (Knowledge distillation) —[improves]→ (large teacher model)
- _16 ms, 3 triples_

**27.** Self-consistency decodes multiple reasoning paths and takes a majority vote, raising GSM8K accuracy of PaLM-540B from 56% to 74%.

- (Self-consistency) —[uses method]→ (multiple reasoning paths)
- (PaLM-540B) —[achieves metric]→ (accuracy)
- (PaLM-540B) —[achieves metric]→ (56%)
- (PaLM-540B) —[achieves metric]→ (74%)
- _18 ms, 4 triples_

## gliner-relex high 0.5/0.65/0.9 (device=cuda)

**1.** Low-Rank Adaptation, or LoRA, reduces hallucination in large language models by constraining weight updates to low-rank matrices.

- (Low-Rank Adaptation) —[reduces]→ (hallucination)
- _16 ms, 1 triples_

**2.** We fine-tune LLaMA-7B on the Alpaca dataset using LoRA and evaluate on MMLU.

- (LLaMA-7B) —[fine-tuned on]→ (Alpaca)
- _15 ms, 1 triples_

**3.** GPT-4 outperforms GPT-3.5 on most benchmarks, achieving 86.4% on MMLU.

- (GPT-4) —[outperforms]→ (GPT-3.5)
- (GPT-4) —[evaluated on]→ (MMLU)
- _16 ms, 2 triples_

**4.** The attention mechanism introduced by Vaswani et al. replaces recurrence entirely with self-attention.

- (attention mechanism) —[reduces]→ (recurrence)
- _16 ms, 1 triples_

**5.** BERT was pretrained on the BookCorpus and English Wikipedia using masked language modeling.

- (BERT) —[pretrained on]→ (BookCorpus)
- (BERT) —[uses method]→ (masked language modeling)
- _16 ms, 2 triples_

**6.** RLHF aligns language model behavior with human preferences by optimizing against a learned reward model.

- (RLHF) —[improves]→ (language model behavior)
- _16 ms, 1 triples_

**7.** InstructGPT demonstrates that reinforcement learning from human feedback improves truthfulness of GPT-3 outputs.

- (reinforcement learning) —[improves]→ (truthfulness)
- _16 ms, 1 triples_

**8.** Chain-of-thought prompting enables large language models to solve multi-step arithmetic reasoning problems.

- (ไม่ได้ triple)
- _17 ms, 0 triples_

**9.** We achieve 94.8 F1 on SQuAD 2.0 using RoBERTa with a span prediction head.

- (ไม่ได้ triple)
- _17 ms, 0 triples_

**10.** FlashAttention reduces memory usage of attention computation from quadratic to linear in sequence length.

- (FlashAttention) —[reduces]→ (memory usage)
- _16 ms, 1 triples_

**11.** Llama 2 introduces a family of models trained on 2 trillion tokens with grouped-query attention.

- (Llama 2) —[introduces]→ (family of models)
- (family of models) —[pretrained on]→ (2 trillion tokens)
- _16 ms, 2 triples_

**12.** Retrieval-augmented generation grounds model answers in documents retrieved by DPR from Wikipedia.

- (ไม่ได้ triple)
- _15 ms, 0 triples_

**13.** Mistral-7B uses sliding window attention and outperforms Llama 2 13B on many tasks.

- (Mistral-7B) —[uses method]→ (sliding window attention)
- (Mistral-7B) —[outperforms]→ (Llama 2 13B)
- _15 ms, 2 triples_

**14.** Qwen2.5 was evaluated on GSM8K, HumanEval, and MMLU, surpassing Qwen2 across all metrics.

- (Qwen2.5) —[evaluated on]→ (GSM8K)
- _16 ms, 1 triples_

**15.** Speculative decoding accelerates inference by drafting tokens with a smaller model and verifying them with the target model.

- (Speculative decoding) —[improves]→ (inference)
- _16 ms, 1 triples_

**16.** The T5 model treats all NLP tasks as text-to-text problems using the span corruption objective.

- (T5 model) —[uses method]→ (NLP)
- _16 ms, 1 triples_

**17.** DeepSeek-R1 distills reasoning ability from large reinforcement-learned models into dense models of 1.5B to 70B parameters.

- (DeepSeek-R1) —[reduces]→ (reasoning ability)
- (DeepSeek-R1) —[improves]→ (reasoning ability)
- _16 ms, 2 triples_

**18.** We report that LoRA matches full fine-tuning performance on GLUE while training 10,000 times fewer parameters.

- (LoRA) —[uses method]→ (full fine-tuning)
- _16 ms, 1 triples_

**19.** LoRA fine-tunes only a small subset of parameters by adding trainable rank decomposition matrices to each transformer layer.

- (ไม่ได้ triple)
- _16 ms, 0 triples_

**20.** Quantization to 4-bit precision reduces GPU memory requirements of large language models with minimal accuracy loss.

- (Quantization) —[reduces]→ (GPU memory requirements)
- _16 ms, 1 triples_

**21.** Direct preference optimization fine-tunes the policy directly on preference data without an explicit reward model.

- (Direct preference optimization) —[improves]→ (policy)
- _15 ms, 1 triples_

**22.** Scaling laws show that loss decreases predictably as model size, dataset size, and compute increase together.

- (ไม่ได้ triple)
- _15 ms, 0 triples_

**23.** DeBERTa improves upon RoBERTa using disentangled attention and achieves 88.8% accuracy on MNLI.

- (DeBERTa) —[uses method]→ (disentangled attention)
- _16 ms, 1 triples_

**24.** BART combines bidirectional encoding with autoregressive generation, pretrained on text infilling objectives.

- (BART) —[uses method]→ (bidirectional encoding)
- (BART) —[introduces]→ (bidirectional encoding)
- (BART) —[uses method]→ (autoregressive generation)
- (BART) —[introduces]→ (autoregressive generation)
- _16 ms, 4 triples_

**25.** PaLM was trained on 780 billion tokens of high-quality text using the Pathways system across 6144 TPU v4 chips.

- (PaLM) —[uses method]→ (Pathways system)
- _17 ms, 1 triples_

**26.** Knowledge distillation transfers the behavior of a large teacher model into a smaller student model such as DistilBERT.

- (ไม่ได้ triple)
- _17 ms, 0 triples_

**27.** Self-consistency decodes multiple reasoning paths and takes a majority vote, raising GSM8K accuracy of PaLM-540B from 56% to 74%.

- (Self-consistency) —[uses method]→ (multiple reasoning paths)
- _18 ms, 1 triples_

## nuextract (device=cuda)

**1.** Low-Rank Adaptation, or LoRA, reduces hallucination in large language models by constraining weight updates to low-rank matrices.

- (LoRA) —[reduces]→ (hallucination in large language models)
- _477 ms, 1 triples_

**2.** We fine-tune LLaMA-7B on the Alpaca dataset using LoRA and evaluate on MMLU.

- (We) —[fine-tune]→ (LLaMA-7B)
- (on) —[the]→ (Alpaca dataset)
- (and) —[evaluate]→ (MMLU)
- _1220 ms, 3 triples_

**3.** GPT-4 outperforms GPT-3.5 on most benchmarks, achieving 86.4% on MMLU.

- (GPT-4) —[outperforms]→ (GPT-3.5)
- _456 ms, 1 triples_

**4.** The attention mechanism introduced by Vaswani et al. replaces recurrence entirely with self-attention.

- (attention mechanism) —[introduced]→ (by Vaswani et al.)
- _437 ms, 1 triples_

**5.** BERT was pretrained on the BookCorpus and English Wikipedia using masked language modeling.

- (BERT) —[was pretrained]→ (on the BookCorpus and English Wikipedia)
- _444 ms, 1 triples_

**6.** RLHF aligns language model behavior with human preferences by optimizing against a learned reward model.

- (RLHF) —[aligns]→ (language model behavior with human preferences)
- _425 ms, 1 triples_

**7.** InstructGPT demonstrates that reinforcement learning from human feedback improves truthfulness of GPT-3 outputs.

- (InstructGPT) —[demonstrates]→ (reinforcement learning from human feedback improves truthfulness of GPT-3 outputs)
- _560 ms, 1 triples_

**8.** Chain-of-thought prompting enables large language models to solve multi-step arithmetic reasoning problems.

- (Chain-of-thought) —[prompting]→ (enables)
- _410 ms, 1 triples_

**9.** We achieve 94.8 F1 on SQuAD 2.0 using RoBERTa with a span prediction head.

- (We achieve) —[94.8 F1]→ (on SQuAD 2.0 using RoBERTa with a span prediction head)
- _569 ms, 1 triples_

**10.** FlashAttention reduces memory usage of attention computation from quadratic to linear in sequence length.

- (FlashAttention) —[reduces]→ (memory usage of attention computation)
- _428 ms, 1 triples_

**11.** Llama 2 introduces a family of models trained on 2 trillion tokens with grouped-query attention.

- (ไม่ได้ triple)
- _304 ms, 0 triples_

**12.** Retrieval-augmented generation grounds model answers in documents retrieved by DPR from Wikipedia.

- (Retrieval-augmented generation) —[ground]→ (model)
- _420 ms, 1 triples_

**13.** Mistral-7B uses sliding window attention and outperforms Llama 2 13B on many tasks.

- (Mistral-7B) —[uses]→ (sliding window attention)
- (Llama 2 13B) —[outperforms]→ (on many tasks)
- _877 ms, 2 triples_

**14.** Qwen2.5 was evaluated on GSM8K, HumanEval, and MMLU, surpassing Qwen2 across all metrics.

- (Qwen2.5) —[was]→ (evaluated)
- (GSM8K) —[was]→ (evaluated)
- (HumanEval) —[was]→ (evaluated)
- (MMLU) —[was]→ (evaluated)
- _1209 ms, 4 triples_

**15.** Speculative decoding accelerates inference by drafting tokens with a smaller model and verifying them with the target model.

- (ไม่ได้ triple)
- _302 ms, 0 triples_

**16.** The T5 model treats all NLP tasks as text-to-text problems using the span corruption objective.

- (ไม่ได้ triple)
- _301 ms, 0 triples_

**17.** DeepSeek-R1 distills reasoning ability from large reinforcement-learned models into dense models of 1.5B to 70B parameters.

- (DeepSeek-R1) —[distills]→ (reasoning ability from large reinforcement-learned models into dense models of 1.5B to 70B parameters)
- _625 ms, 1 triples_

**18.** We report that LoRA matches full fine-tuning performance on GLUE while training 10,000 times fewer parameters.

- (LoRA) —[matches]→ (full fine-tuning performance on GLUE)
- (GLUE) —[training]→ (10,000 times fewer parameters)
- _761 ms, 2 triples_

**19.** LoRA fine-tunes only a small subset of parameters by adding trainable rank decomposition matrices to each transformer layer.

- (LoRA) —[fine-tunes]→ (only a small subset of parameters)
- _437 ms, 1 triples_

**20.** Quantization to 4-bit precision reduces GPU memory requirements of large language models with minimal accuracy loss.

- (Quantization to 4-bit precision) —[reduces]→ (GPU memory requirements)
- _456 ms, 1 triples_

**21.** Direct preference optimization fine-tunes the policy directly on preference data without an explicit reward model.

- (Direct preference optimization) —[fine-tunes]→ (the policy directly on preference data)
- _450 ms, 1 triples_

**22.** Scaling laws show that loss decreases predictably as model size, dataset size, and compute increase together.

- (loss) —[decreases]→ (predictably)
- _386 ms, 1 triples_

**23.** DeBERTa improves upon RoBERTa using disentangled attention and achieves 88.8% accuracy on MNLI.

- (DeBERTa) —[improves upon]→ (RoBERTa)
- (MNLI) —[achieves]→ (88.8%)
- _712 ms, 2 triples_

**24.** BART combines bidirectional encoding with autoregressive generation, pretrained on text infilling objectives.

- (ไม่ได้ triple)
- _301 ms, 0 triples_

**25.** PaLM was trained on 780 billion tokens of high-quality text using the Pathways system across 6144 TPU v4 chips.

- (PaLM) —[was trained]→ (on 780 billion tokens of high-quality text)
- _474 ms, 1 triples_

**26.** Knowledge distillation transfers the behavior of a large teacher model into a smaller student model such as DistilBERT.

- (Knowledge) —[transfers]→ (the behavior of a large teacher model into a smaller student model)
- _472 ms, 1 triples_

**27.** Self-consistency decodes multiple reasoning paths and takes a majority vote, raising GSM8K accuracy of PaLM-540B from 56% to 74%.

- (ไม่ได้ triple)
- _302 ms, 0 triples_

## สรุปเบื้องต้น

| config | triples รวม |
|---|---|
| gliner-relex lo 0.3/0.5/0.5 | 130 |
| gliner-relex mid 0.5/0.6/0.7 | 73 |
| gliner-relex high 0.5/0.65/0.9 | 29 |
| nuextract | 30 |