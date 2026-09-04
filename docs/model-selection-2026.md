# Model selection สำหรับ triple extraction จาก paper วิชาการ (สถานะ ก.ย. 2026)

## โจทย์สรุป

- Input: text จาก OCR service ภายนอก (มี markdown table, บาง pipeline ให้มาเป็น HTML table, มี LaTeX สมการ)
- Domain: วิศวกรรม / AI / LLM / computer science
- Schema: **ยังไม่มี relation list ที่ชัดเจน** → ต้อง open schema หรืออัปเดต relation list ได้แบบอัตโนมัติ
- คำถามย่อย: model อ่าน markdown table / HTML table / LaTeX ได้ไหม
- รูปภาพ: ยังไม่ต้องจัดการ

> หมายเหตุ: ใน repo นี้ REBEL และ NuExtract (gen 1) ถูกทดสอบแล้วและตัดออก GLiNER-Relex เป็น default ปัจจุบัน (ดู `docs/bakeoff-results.md`, `docs/bakeoff-accuracy-report.md`) — รายงานนี้มองกว้างกว่านั้นสำหรับเลือกตัวที่จะ bake-off ต่อ

## กลุ่มที่ 1: Dedicated Information Extraction (Specialized IE Models)

model ที่สร้าง/finetune มาเพื่องาน extraction โดยเฉพาะ ส่วนใหญ่เป็น encoder เล็ก เร็ว ถูก แต่อ่านโครงสร้างพิเศษ (table, LaTeX) ไม่ได้

| Model | ปีที่ออก | Size | รายละเอียด | Open schema | Markdown/HTML table | LaTeX | คะแนนที่เกี่ยวข้อง | ทรัพยากรโดยประมาณ |
|---|---|---|---|---|---|---|---|---|
| **GLiNER-Relex** (knowledgator/gliner-relex-large-v1.0) | 2024 | ~435M (DeBERTa-v3-large) | Encoder, zero-shot NER+RE ใน pass เดียว — default ปัจจุบันของ repo นี้ | ✅ (ใส่ label list เป็น hint ได้) | ❌ ต้อง preprocess (linearize table เป็นประโยค) | ❌ ต้อง strip/replace สมการ | GLiNER multi-task รายงาน ~82.5 EM / 87.4 F1 บน RE benchmark ทั่วไป | CPU ได้, GPU ~1-2GB VRAM, ms/ประโยค (เครื่องเราวัดได้ 18-155 ms) |
| **GLiREL** (jackboyla/glirel-large-v0) | 2025 (NAACL) | ~467M (DeBERTa-v3-large) | Encoder, zero-shot RE จาก entity ที่ให้มา (ต้องมี NER ตั้งต้น) | ✅ | ❌ | ❌ | FewRel 94.20 F1 (ชนะ GPT-4o ใน setting m=5), Wiki-ZSL 83.28 F1 | CPU ได้, GPU ~1-2GB VRAM |
| **GLiNER 2.x / multi-task** (knowledgator) | 2025 | ~300-900M ตาม checkpoint | Encoder, schema-driven extraction (entity/relation/โครงสร้างซ้อน) ใน pass เดียว | ✅ ออกแบบมาให้ schema เป็น input | ⚠️ พอ preprocess แล้ว | ⚠️ | ~87.4 F1 RE (multi-task variant) | CPU/GPU เล็ก |
| **SciER fine-tuned** (baseline จาก dataset paper, EMNLP 2024) | 2024 | ~110M-400M | Encoder fine-tune บน scientific NER+RE (methods/tasks/metrics, uses/extends/compares) | ❌ schema ตายตัวตาม dataset | ❌ | ❌ | SciER test: F1 ระดับ 70-80 สำหรับ RE บน abstract วิทยาศาสตร์ (baseline แนว BERT/SciBERT) | CPU/GPU เล็กมาก |
| **NuExtract 2.0** (numind) | 2025 | 0.5B-8B (fine-tune จาก Qwen2.5) | Decoder แต่ train มาเพื่อ text→JSON โดยเฉพาะ — จัดเป็น dedicated IE แม้สถาปัตยกรรมเป็น LLM | ✅ ผ่าน JSON template | ⚠️ อ่าน markdown ได้ระดับหนึ่ง แต่ fine-tune มาบนข้อมูลสั้น ๆ ไม่ใช่ paper เต็ม | ⚠️ | ไม่มี benchmark สาธารณะด้าน scientific RE ที่ชัดเจน | 4B ~8-10GB VRAM fp16, quantize แล้ว ~3GB |
| **ReLiK** (sapienzanlp/relik-relation-extraction-large) | 2024 (Findings ACL) | 183M (base) / 434M (large), DeBERTa-v3 | Retriever-Reader ทำ Entity Linking + RE พร้อมกัน ออกแบบมาให้เบาและเร็ว มี integration สำเร็จรูปกับ LlamaIndex/Neo4j สำหรับ KG construction | ⚠️ relation จำกัดตาม knowledge base (Wikipedia) — ไม่ใช่ open schema เต็ม | ❌ | ❌ | ชนะ baseline EL/RE ทั่วไปด้วย compute น้อยกว่ามาก ถูกใช้เป็นตัว extraction หลักใน KG pipeline จริง | CPU ได้, GPU ~1-2GB VRAM |
| **GoLLIE** (HiTZ/GoLLIE-7B/13B) | 2023 (CVPR 2024) | 7B / 13B (Code Llama fine-tune) | fine-tune ให้ "อ่าน annotation guideline" แล้วทำ IE ตาม guideline ที่เขียนเป็น Python class ได้ทันทีตอน inference | ✅ กำหนด schema ใหม่ได้ด้วยการเขียน guideline | ⚠️ ฐานเป็น code-LLM อ่าน markdown/LaTeX ได้พอควร แต่ถูก tune บน IE data สั้น ๆ | ⚠️ | zero-shot IE บน unseen schema ชนะ baseline ยุคนั้นหลาย benchmark (ACE, SciERC รวมอยู่) | 7B ~5-6GB Q4, 13B ~9GB Q4 |
| **KnowCoder** | 2024 (ACL) | ~7B (Llama-based) | Unified IE ผ่าน code generation — แปลง text เป็น Python object ที่มีโครงสร้าง (entity/relation/event) tune บนข้อมูล IE ~800K instances | ✅ ผ่าน code schema | ⚠️ | ⚠️ | ชนะ GoLLIE/USM บนหลาย benchmark IE (รวม SciERC) | ~5-6GB Q4 |
| **InstructUIE** | 2023 | Flan-T5 (250M-11B) | instruction tuning รวม NER+RE+EE เป็น task เดียว | ✅ ผ่าน instruction | ❌ (encoder-decoder อ่อนด้าน table/LaTeX) | ❌ | แข่งใกล้เคียง fine-tune เฉพาะ task บน IE INSTRUCTIONS แต่เก่าและไม่มี update ต่อ | เล็ก-กลาง, GPU ไม่กี่ GB |
| **CodeKGC** | 2023 (ACM TOIS 2024) | ขึ้นกับ code-LLM ที่เอามาใช้ (CodeLlama 7B เป็นต้น) | KGC เป็น code generation + schema-aware prompt + rationale generation | ✅ ผ่าน schema ใน prompt | ⚠️ | ⚠️ | ชนะ baseline generative KGC บน NYT/WebNLG | ตาม base model (~5GB Q4 ที่ 7B) |

ข้อดีกลุ่มนี้: เร็ว (ms/ประโยค), VRAM น้อย, output ตรง format ไม่ต้องกลัว hallucinate schema ข้อเสีย: โลกของมันคือประโยคธรรมชาติ — markdown table, HTML, LaTeX ต้องผ่าน preprocess ก่อนเสมอ (ยกเว้นตระกูล code-LLM อย่าง GoLLIE/KnowCoder/CodeKGC ที่ทนสักหน่อย แต่ก็ tune มาบน IE data สั้น ไม่ใช่ paper เต็ม) อีกข้อสังเกต: พวก UIE/code-based (GoLLIE, KnowCoder, InstructUIE, CodeKGC) คือ "dedicated IE ที่เป็น LLM จริง ๆ" — เก่าทั้งหมด (2023-2024) และถูกกลืนโดย general LLM รุ่นใหม่ในแง่คุณภาพ แต่ยังเก่งกว่าเรื่อง output format ที่รัดกว่า

## กลุ่มที่ 2: LLM-based (General LLM ใช้ผ่าน prompting)

model ทั่วไปที่สั่งงานด้วย prompt ให้คืน triple เป็น JSON — open schema ได้เต็มที่ อ่านโครงสร้างพิเศษได้ในตัว แลกกับความเร็วและ VRAM

| Model | ปีที่ออก | Size | Open schema | Markdown/HTML table | LaTeX | คะแนนที่เกี่ยวข้อง | ทรัพยากรโดยประมาณ |
|---|---|---|---|---|---|---|---|
| **Qwen3** (4B/8B/14B/30B-A3B) | 2025 (มี update 2507) | 0.6B-235B | ✅ prompt ได้ทุก schema | ✅ **เข้าใจ markdown/HTML table เป็น native** | ✅ **อ่าน LaTeX ได้ดีที่สุดในกลุ่ม open-weight** | benchmark GraphRAG extraction ปี 2026: ตระกูล Qwen ได้คุณภาพ graph ดีที่สุดในกลุ่ม local; 4B+ ชนะ MMLU/GSM8K รุ่นเดียวกันเกือบทุกตัว | 4B ~8GB VRAM fp16 / ~3GB Q4, 8B ~16GB fp16 / ~5-6GB Q4, 14B ~10GB Q4 |
| **Llama 3.1 8B / Llama 4 Scout** | 2024 / 2025 | 8B / 109B-MoE (17B active) | ✅ | ✅ | ✅ (แต่ LaTeX อ่อนกว่า Qwen3 เล็กน้อย) | re-benchmark GraphRAG กลางปี 2026: Llama 3.1 8B ให้ entity/relation เยอะที่สุด (1,172 entities / 696 relations ต่อ corpus เดียวกัน) แต่ index ช้าสุด (211 นาที) | 8B ~16GB fp16 / ~5-6GB Q4 |
| **Gemma 3** (4B/12B/27B) | 2025 | 1B-27B | ✅ | ✅ | ✅ | LMArena ตอนเปิดตัวติด top ของ open-weight; ไม่มี benchmark RE เฉพาะทาง | 12B ~8GB Q4, 27B ~17GB Q4 |
| **Phi-4 / Phi-4-mini** | 2024/2025 | 14B / 3.8B | ✅ | ✅ | ✅ (แข็งด้าน math) | คะแนน math/reasoning สูงผิดขนาด แต่ instruction-following สำหรับ structured output อ่อนกว่า Qwen3 | 3.8B ~3GB Q4, 14B ~9GB Q4 |

หมายเหตุ: **Dagdelen et al. / Lagrange** (Nature Comms 2024, fine-tuned decoder สำหรับ scientific NER+RE, ~125M-7B) อยู่ตรงกลางระหว่างสองกลุ่ม — สถาปัตยกรรม LLM แต่ fine-tune ลง task เดียว schema ตายตัว และเน้นวัสดุศาสตร์ ไม่เข้าข่าย bake-off ของเรา

## ตอบคำถามย่อย: table กับ LaTeX

- **กลุ่ม Dedicated IE: อ่าน markdown/HTML table และ LaTeX ตรง ๆ ไม่ได้** — train มาบนประโยคธรรมชาติ การเอา `| col | col |` หรือ `$\alpha_{ij}$` เข้าไปจะได้ triple ขยะหรือหลุดทั้งก้อน ต้อง preprocess: linearize table เป็นประโยค ("Model A achieves 86.4% on MMLU"), แปลงสมการเป็นชื่อสัญลักษณ์หรือข้ามไป (ยกเว้น NuExtract 2.0 ที่ทน markdown ได้บ้าง)
- **กลุ่ม LLM: อ่าน markdown table ได้ดี, HTML ได้ดีกว่า, LaTeX อ่านได้** — Qwen3 กับ Phi-4 แข็งแรงสุดด้านสัญลักษณ์ math เพราะ pretrain data มี LaTeX เยอะ ข้อจำกัดจริงคือ context length ต่อ chunk และ inference ต่อ token แพงกว่า encoder ~100 เท่า
- **สมการ: ไม่มี model ใด "แยก relation จากสมการ" โดยตรง** — ที่ทำได้จริงคืออ่านสมการเป็นสัญลักษณ์แล้วสกัดความสัมพันธ์แบบ "X นิยามด้วย/คำนวณจาก Y" ซึ่งเป็นงาน prompt engineering บน LLM ไม่ใช่ความสามารถ in-the-weights

## แนะนำ 1-5 อันดับที่ควรเอามา bake-off

1. **Qwen3-8B (หรือ 4B ถ้า VRAM ติด)** [LLM] — ตัวเดียวที่ครอบคลุมโจทย์ครบ: open schema ผ่าน prompt, อ่าน markdown/HTML table และ LaTeX ได้โดยไม่ต้อง preprocess, คุณภาพ KG extraction จาก benchmark local ปี 2026 ดีที่สุดในกลุ่ม open-weight ขนาดพอดี GPU เดียว Q4 ใช้ ~5-6GB ข้อแลก: ช้ากว่า encoder มาก และยังต้องพิสูจน์ว่า instruction-following สำหรับ JSON รัดกว่า GLiNER-Relex จริง
2. **GLiNER-Relex (คงไว้เป็น baseline)** [Dedicated IE] — เร็วระดับ ms, VRAM น้อยสุด, เครื่องเรามี bake-off ผลล่าสุดแล้ว เหมาะเป็น "งบต่ำ/ปริมาณสูง" path และเป็นตัวเทียบว่า LLM คุ้มค่าไหม
3. **GLiREL + GLiNER (คู่)** [Dedicated IE] — สถานะศิลป์ของ encoder zero-shot RE (FewRel 94.2 F1) ลองดูว่าแยก entity ด้วย GLiNER แล้วส่งให้ GLiREL จะได้ precision สูงกว่า Relex หรือไม่ เหมาะถ้าเรายอม preprocess table/LaTeX
4. **Gemma 3 12B** [LLM] — LLM ตัวที่สองเพื่อเทียบ bias ของ Qwen, multimodal ในตัว (เผื่อวันหน้าเอารูป/figure เข้า pipeline ได้เลย) Q4 ใช้ ~8GB
5. **Phi-4-mini (3.8B)** [LLM] — ตัวเลือกถ้าอยากได้ LLM ที่เบามากแต่เก่ง math/LaTeX ไว้ลองเป็น tier ล่างของ LLM path

สรุปทิศทาง: **จริงจังกับ table + LaTeX แล้วต้องเดินสาย LLM (Qwen3 เป็นตัวตั้งต้น)** — Dedicated IE path ต้องบวกค่า preprocess ที่จะพังบ่อยกับข้อมูลจริงจาก OCR ที่เหมาะสุดคือ hybrid: LLM สำหรับส่วน table/equation-heavy, Dedicated IE สำหรับ text ที่เหลือเมื่อปริมาณงานใหญ่

## Sources

- [GLiREL paper (NAACL 2025, arXiv:2501.03172)](https://arxiv.org/html/2501.03172v1) · [GitHub](https://github.com/jackboyla/GLiREL)
- [GLiNER multi-task (arXiv:2406.12925)](https://arxiv.org/html/2406.12925v2) · [gliner-relex-large-v1.0](https://huggingface.co/knowledgator/gliner-relex-large-v1.0)
- [GLiNER-BioMed (2025/2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13259603/)
- [SciER dataset (arXiv:2410.21155)](https://arxiv.org/html/2410.21155v1)
- [SciEx: LLM framework สำหรับ scientific IE (arXiv:2512.10004)](https://arxiv.org/html/2512.10004v1)
- [NuExtract 2.0 1B](https://huggingface.co/numind/NuExtract-2-1B-experimental) · [MeXtract อ้างขนาด NuExtract 0.5B-3B](https://arxiv.org/html/2510.06889v1)
- [Local LLMs for Graph RAG extraction: mid-2026 re-benchmark](https://medium.com/@shereshevsky/local-llms-for-graph-rag-extraction-the-mid-2026-re-benchmark-5f36b3d19383)
- [Benchmark document parsers บนเนื้อหา math (arXiv:2512.09874)](https://arxiv.org/html/2512.09874v2)
- [Open Local KG Construction from Academic Papers (ACM 2025)](https://dl.acm.org/doi/10.1145/3701716.3717820)
- [ReLiK: Retrieve and LinK (arXiv:2408.00103)](https://arxiv.org/abs/2408.00103) · [sapienzanlp/relik-relation-extraction-large](https://huggingface.co/relik-ie/relik-relation-extraction-large)
- [GoLLIE (arXiv:2310.03668)](https://arxiv.org/abs/2310.03668) · [HiTZ/GoLLIE-13B](https://huggingface.co/HiTZ/GoLLIE-13B)
- [InstructUIE (arXiv:2304.08085)](https://www.alphaxiv.org/abs/2304.08085)
- [YAYI-UIE (arXiv:2312.15548)](https://arxiv.org/abs/2312.15548)
- [CodeKGC (ACM TOIS)](https://dl.acm.org/doi/10.1145/3641850)
