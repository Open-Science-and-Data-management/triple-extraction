# **รายงานการศึกษาวิจัย: เกณฑ์มาตรฐานและสถาปัตยกรรมโมเดลภาษาขนาดเล็กสำหรับการสกัดความสัมพันธ์เชิงสามส่วน**

การสกัดความสัมพันธ์เชิงสามส่วน (Relation Triplet Extraction หรือ RTE) เป็นหมุดหมายสำคัญในสาขาการประมวลผลภาษาธรรมชาติ (Natural Language Processing: NLP) และวิทยาการปัญญาประดิษฐ์ กระบวนการนี้มีวัตถุประสงค์ในการแปลงข้อความภาษาธรรมชาติที่ปราศจากโครงสร้าง ให้กลายเป็นโครงสร้างสารสนเทศเชิงสัมพันธ์ที่ประกอบด้วย ประธาน ภาคแสดง และกรรม หรือ (Subject, Relation, Object) โครงสร้างข้อมูลสามส่วนดังกล่าวถือเป็นองค์ประกอบรากฐานในการสร้างกราฟความรู้ (Knowledge Graph Construction) การตอบคำถามเชิงความหมาย (Question Answering) ตลอดจนการค้นคืนสารสนเทศขั้นสูง  
แม้ว่าโมเดลภาษาขนาดใหญ่ (Large Language Models: LLMs) จะแสดงความสามารถในการทำความเข้าใจภาษาในระดับสูง แต่การประยุกต์ใช้ LLMs ในภารกิจการสกัดความสัมพันธ์เชิงสามส่วนกลับประสบข้อจำกัดด้านต้นทุนการประมวลผล ความล่าช้าในการตอบสนอง และผลกระทบต่อสิ่งแวดล้อมจากการใช้พลังงานไฟฟ้าในปริมาณสูง นอกจากนี้ งานวิจัยยังพบว่า LLMs มักประสบปัญหาอัตราการดึงกลับของข้อมูลต่ำ (Low Recall) เมื่อต้องจัดการกับข้อความที่มีความยาวและซับซ้อน เนื่องจากโมเดลมีแนวโน้มที่จะตกหล่นข้อมูลสามส่วนที่มีความเกี่ยวเนื่องกันหลายประการในประโยคเดียวกัน  
ด้วยเหตุนี้ โมเดลภาษาขนาดเล็ก (Small Language Models: SLMs) ซึ่งมีขนาดพารามิเตอร์ตั้งแต่ระดับต่ำกว่า 1 พันล้านตัวไปจนถึงประมาณ 7 พันล้านตัว จึงได้รับการพัฒนาและประเมินประสิทธิภาพอย่างเป็นระบบในฐานะทางเลือกทางเทคโนโลยีที่มีความยั่งยืน มีประสิทธิภาพการคำนวณสูง และสามารถติดตั้งใช้งานบนระบบที่มีทรัพยากรจำกัดได้อย่างมีประสิทธิผล

## **สถาปัตยกรรมและกลไกเชิงทฤษฎีของ SLM สำหรับการสกัด Triplet**

การพัฒนาระบบสกัดความสัมพันธ์เชิงสามส่วนด้วยโมเดลภาษาขนาดเล็กได้วิวัฒนาการผ่านหลายกระบวนทัศน์ทางสถาปัตยกรรม เพื่อข้ามผ่านข้อจำกัดด้านความเร็วในการประมวลผลและปัญหาการส่งผ่านข้อผิดพลาดเชิงลำดับ

### **สถาปัตยกรรมสร้างลำดับต่อลำดับ (Autoregressive Seq2Seq Framework)**

การเปลี่ยนผ่านจากการสกัดแบบหลายขั้นตอน (Multi-step Pipelines) มาสู่ระบบ End-to-End เริ่มต้นจากการใช้โมเดลประเภท Sequence-to-Sequence (Seq2Seq) เช่น [REBEL (EMNLP 2021\)](https://github.com/Babelscape/rebel) ซึ่งพัฒนาขึ้นบนสถาปัตยกรรม BART-large REBEL ทำการแปลงตารางความสัมพันธ์เชิงสามส่วนให้อยู่ในรูปแบบเชิงเส้น (Linearization) โดยใช้โทเค็นพิเศษในการแบ่งแยก Subject, Object และ Relation Type ทำให้โมเดลสามารถสกัดความสัมพันธ์ได้มากกว่า 200 ประเภทในขั้นตอนเดียว การฝึกฝนโมเดลกระทำผ่านชุดข้อมูลขนาดใหญ่ที่กรองด้วยเทคนิค Natural Language Inference (NLI) บนคลังข้อมูล [REBEL Dataset (HuggingFace)](https://huggingface.co/datasets/Babelscape/rebel-dataset) ซึ่งมีข้อมูลมากกว่า 3.47 ล้านแถว อย่างไรก็ตาม ข้อจำกัดทางสถาปัตยกรรมแบบสร้างลำดับคำทีละโทเค็น (Autoregressive Decoding) ส่งผลให้เกิดคอขวดด้านความเร็วเมื่อต้องประมวลผลเอนทิตีจำนวนมาก

### **สถาปัตยกรรมแบบ Bi-Encoder และการจับคู่เชิงพื้นที่เวกเตอร์ (Open-Vocabulary Matching)**

เพื่อแก้ปัญหาความล่าช้าของโมเดลแบบ Autoregressive ได้มีการเสนอกรอบการทำงาน [GLiREL (arXiv:2501.03172)](https://arxiv.org/abs/2501.03172) ซึ่งต่อยอดจากสถาปัตยกรรม GLiNER GLiREL ปรับใช้โครงสร้างตัวเข้ารหัสสองทาง (Bidirectional Encoder) ร่วมกับโมดูลการคำนวณเวกเตอร์ของคู่เอนทิตี (Entity Pair Representation) และโมดูลให้คะแนนความคล้ายคลึง (Scorer Module) ระบบนี้สามารถประมวลผลป้ายกำกับความสัมพันธ์แบบเปิด (Zero-shot / Open-vocabulary) ร่วมกับคู่เอนทิตีทั้งหมดในข้อความได้ในการส่งข้อมูลผ่านโมเดลเพียงครั้งเดียว (Single Forward Pass) ช่วยลดภาระการคำนวณและเพิ่มความเร็วในการประมวลผลอย่างมหาศาล  
สำหรับการประมวลผลระดับเอกสาร (Document-level RE) ที่มีความยาวและหนาแน่นด้วยเอนทิตี มีการพัฒนาสถาปัตยกรรม [GLiDRE (arXiv:2508.00757)](https://arxiv.org/abs/2508.00757) ซึ่งประยุกต์ใช้ระบบ Dual-Encoder แยกชุดประมวลผลระหว่างตัวบทและป้ายกำกับความสัมพันธ์ โดยผสานกลไก Localized Context Pooling เพื่อรวบรวมบริบทข้ามประโยคและคำนวณเวกเตอร์ความสัมพันธ์ของคู่เอนทิตีอย่างมีประสิทธิภาพ

### **การย่อยประพจน์เชิงอะตอม (Atomic Proposition Decomposition)**

แนวทาง [MPropositionneur-V2 (arXiv:2604.02866)](https://arxiv.org/abs/2604.02866) นำเสนอเทคนิคการย่อยประโยคภาษาธรรมชาติที่มีความซับซ้อนให้กลายเป็น "ประพจน์เชิงอะตอม" (Atomic Propositions) ซึ่งเป็นหน่วยสารสนเทศย่อยที่สุดที่เป็นอิสระเชิงความหมายและสอดคล้องกับหลักไวยากรณ์ Conjunctive Normal Form (CNF) MPropositionneur-V2 ใช้สถาปัตยกรรม Qwen3-0.6B ที่ผ่านการกลั่นกรองความรู้ (Knowledge Distillation) จาก Qwen3-32B ทำหน้าที่เป็นตัวเตรียมข้อมูลแบบหลายภาษา ก่อนจะส่งต่อข้อความย่อยให้ตัวสกัดอย่าง GLiREL หรือ Qwen3-4B ซึ่งผลการทดลองยืนยันว่ากลยุทธ์นี้ช่วยเพิ่มค่า Recall ของความสัมพันธ์ในประโยคที่มีโครงสร้างซับซ้อนได้อย่างมีนัยสำคัญ

### **สถาปัตยกรรมกำกับสกีมาและระบบที่อธิบายผลลัพธ์ได้**

โครงสร้างการสกัดความสัมพันธ์ยังครอบคลุมถึงระบบ [SMARTe (arXiv:2504.12816)](https://arxiv.org/abs/2504.12816) ซึ่งนำเสนอกลไก Slot Attention เพื่อระบุโทเค็นที่ส่งผลต่อการตัดสินใจสกัด triplet ทำให้สร้างคำอธิบายความสัมพันธ์ได้อย่างโปร่งใส ขณะที่ระบบ [OneKE (ZJU-NLP / Ant Group)](https://huggingface.co/zjunlp/OneKE) ได้ถูกพัฒนาให้เป็นกรอบการทำงานแบบ Schema-Guided สำหรับภาษาจีนและภาษาอังกฤษ ซึ่งบังคับให้การสกัด triplet เป็นไปตามสกีมาหรือโดเมนออนโทโลจีที่กำหนดไว้ได้อย่างเข้มงวด

| สถาปัตยกรรม / โมเดล | กระบวนทัศน์การประมวลผล | จุดเด่นทางเทคนิค | ลิงก์เข้าถึงเนื้อหาต้นฉบับ |
| :---- | :---- | :---- | :---- |
| REBEL (BART-large) | Autoregressive Seq2Seq | แปลงภารกิจเป็น Seq2Seq โดยใช้โทเค็นพิเศษกำกับ triplet สกัดความสัมพันธ์ได้ครอบคลุม | [REBEL GitHub / Model](https://github.com/Babelscape/rebel) |
| GLiREL | Bi-Encoder Open-Vocabulary | ประมวลผล Label และ Entity Pairs ใน Forward Pass เดียว สนับสนุน Zero-shot | [GLiREL Paper (arXiv:2501.03172)](https://arxiv.org/abs/2501.03172) |
| GLiDRE | Bi-Encoder Document-Level | ปรับใช้ Localized Context Pooling เพื่อสกัดความสัมพันธ์ระดับเอกสาร | [GLiDRE Paper (arXiv:2508.00757)](https://arxiv.org/abs/2508.00757) |
| MPropositionneur-V2 | Atomic Proposition Distillation | สกัดประพจน์อะตอมตามหลัก CNF ด้วย Qwen3-0.6B เพื่อเพิ่ม Recall | MPropositionneur-V2 (arXiv:2604.02866) |
| SMARTe | Slot-based Attention | เพิ่มความโปร่งใสของกระบวนการสกัด triplet ผ่าน Slot-based token absorption | SMARTe Paper (arXiv:2504.12816) |
| OneKE | Schema-Guided Generative | สกัดข้อมูลตามสกีมาอย่างเคร่งครัด สำหรับภาษาจีนและภาษาอังกฤษ | [OneKE Repository (HuggingFace)](https://huggingface.co/zjunlp/OneKE) |

## **การประเมินผลเชิงเปรียบเทียบบนเกณฑ์มาตรฐานระดับสากล**

การทดสอบประสิทธิภาพของ SLM บนเกณฑ์มาตรฐาน (Benchmarks) ได้แบ่งออกตามลักษณะมิติของข้อความและการทดสอบในสภาวะจำกัดทรัพยากร

### **เกณฑ์มาตรฐานระดับประโยค (NYT และ WebNLG Benchmarks)**

ชุดข้อมูล NYT (New York Times) และ WebNLG เป็นเกณฑ์มาตรฐานหลักในการทดสอบการสกัด triplet ระดับประโยคเดี่ยว จากการศึกษาเปรียบเทียบในงานวิจัย [SMARTe (arXiv:2504.12816)](https://arxiv.org/abs/2504.12816) และ [LLM Triplet Extraction Evaluation (arXiv:2312.01954)](https://arxiv.org/abs/2312.01954) พบว่าโมเดลขนาดเล็กที่ผ่านการ Fine-tune เฉพาะทางทำคะแนน F1-Score ได้สูงกว่าโมเดลภาษาขนาดใหญ่ทั่วไปที่ประเมินในรูปแบบ Zero-shot อย่างชัดเจน

| สถาปัตยกรรม / โมเดล | ประเภทโมเดล / การตั้งค่า | NYT Precision (%) | NYT Recall (%) | NYT F1-Score (%) | WebNLG F1-Score (%) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| UniRel | Supervised Fine-tuned | 93.5 | 94.0 | 93.7 | 94.7 |
| DirectRel | Supervised Fine-tuned | 93.6 | 92.2 | 92.9 | \- |
| SMARTe (Opt Transport) | Supervised Fine-tuned | 92.7 | 93.1 | 92.9 | \- |
| TPLinker | Supervised Fine-tuned | 91.4 | 92.6 | 92.0 | \- |
| Llama3.3-70B | Zero-shot LLM | 7.0 | 15.2 | 8.9 | \- |
| Gemma3-27B | Zero-shot SLM/LLM | 6.0 | 12.0 | 7.3 | \- |
| Qwen3-32B | Zero-shot LLM | 3.7 | 8.2 | 4.7 | \- |
| Phi4-14B | Zero-shot SLM | 3.3 | 6.1 | 4.0 | \- |
| GPT-4o-Mini | Zero-shot LLM | 3.3 | 6.3 | 3.9 | \- |

ผลลัพธ์เชิงตัวเลขสะท้อนให้เห็นว่าโมเดลภาษาขนาดใหญ่ทั่วไปเมื่อนำมาใช้งานแบบ Zero-shot โดยไม่มีการปรับแต่งเฉพาะทาง จะให้ค่า F1-Score ต่ำกว่า 10% บนชุดข้อมูล NYT เนื่องจากโมเดลขาดความเข้าใจในข้อกำหนดขอบเขตของเอนทิตีและการจับคู่ความสัมพันธ์ตามสกีมาของชุดข้อมูลอ้างอิง

### **เกณฑ์มาตรฐานการจัดหมวดหมู่แบบ Zero-Shot (FewRel และ Wiki-ZSL)**

ในการประเมินความสามารถข้ามโดเมนและการจำแนกความสัมพันธ์ที่ไม่เคยพบมาก่อน (Unseen Relations: m) ชุดข้อมูล FewRel และ Wiki-ZSL ได้ถูกนำมาใช้ทดสอบโมเดล GLiREL โดยเปรียบเทียบกับสถาปัตยกรรม Zero-shot RE อื่นๆ

| จำนวนความสัมพันธ์ที่ไม่เคยพบ (m) | สถาปัตยกรรม / โมเดล | Wiki-ZSL F1-Score (%) | FewRel F1-Score (%) | ลิงก์อ้างอิงเนื้อหา |
| :---- | :---- | :---- | :---- | :---- |
| m \= 5 | ZSRE | 95.46 | 96.51 | [GLiREL Paper (arXiv:2501.03172)](https://arxiv.org/abs/2501.03172) |
| m \= 5 | GLiREL (+ synthetic pretraining) | 83.28 | 94.20 | [GLiREL Paper (arXiv:2501.03172)](https://arxiv.org/abs/2501.03172) |
| m \= 5 | TMC-BERT | 88.92 | 93.62 | [GLiREL Paper (arXiv:2501.03172)](https://arxiv.org/a[span_32]\(start_span\)[span_32]\(end_span\)bs/2501.03172) |
| m \= 5 | GPT-4o | 80.03 | 89.20 | \[GLiREL Paper (arXiv:2501.03172)\](https://arxiv.org/abs/2501.03172) |
| m \= 15\[span\_34\](start\_span)\[span\_34\](end\_span) | GLiREL (+ synthetic pretraining) | 73.91 | 84.48 | \[GLiREL Paper (arXiv:2501.03172)\](https://arxiv.org/abs/2501.03172) |
| m \= 15 | TMC-BERT | 73.77 | 81.00 | \[GLiREL Paper (arXiv:2501.03172)\](https://arxiv.org/abs/2501.03172) |
| m \= 15 | DSP-ZRSC | 70.40 | 80.40 | \[GLiREL Paper (arXiv:2501.03172)\](https://arxiv.org/abs/2501.03172) |
| m \[span\_40\](start\_span)\[span\_40\](end\_span)= 15 | GPT-4o | 41.57 | 70.70 | [GLiREL Paper (arXiv:2501.03172)](https://arxiv.org/a[span_41]\(start_span\)[span_41]\(end_span\)bs/2501.03172) |

การวิเคราะห์ข้อมูลชี้ให้เห็นว่าเมื่อจำนวนประเภทความสัมพันธ์ที่ไม่เคยพบมาก่อนเพิ่มขึ้นจาก m=\[span\_43\](start\_span)\[span\_43\](end\_span)5 เป็น m=15 โมเดล GLiREL ที่ผ่านการ Pretrain ด้วยข้อมูลสังเคราะห์ สามารถรักษาสภาพการลดลงของ F1-Score ได้ดีกว่า GPT-4o อย่างชัดเจน โดยเฉพาะในชุดข้อมูล Wiki-ZSL ที่ GPT-4o ประสบปัญหาค่า Recall ตกลงอย่างรุนแรงจนทำให้ F1-Score เหลือเพียง 41.57% ในขณะที่ GLiREL ยังคงทำคะแนนได้ 73.91%

### **เกณฑ์มาตรฐานระดับเอกสาร (Re-DocRED Benchmark)**

การสกัดความสัมพันธ์ในข้อความระดับเอกสารมีความท้าทายสูงจากความซับซ้อนของเอนทิตีที่ปรากฏข้ามประโยค การประเมินบน [Re-DocRED (arXiv:2508.00757)](https://arxiv.org/abs/2508.00757) ได้เปรียบเทียบโมเดลในสถานะ Supervised, Low-Resource (N ตัวอย่าง) และ Zero-shot  
| การตั้งค่าทดลอง | สถาปัตยกรรม / โมเดล | Test F1-Score (%) | Test Ign F1-Score (%) | | \--- | \--- | \--- | \--- | | Fully Supervised | DREEAM | 80.20 | 78.56 | | Fully Supervised | TTM-RE | 79.95 | 78.20 | | Fully Supervised | GLiDRE (Ours) | 77.83 | 76.80 | | Fully Supervised | LMRC LLaMA2-13B-Chat | 74.63 | 74.08 | | Low-Resource (N=10) | GLiDRE (Ours) | 41.73 | \- | | Low-Resource (N=10) | ATLOP | 29.48 | \- | | Low-Resource (N=10) | DREEAM | 27.07 | \- | | Zero-shot | Mistral-Large 123B | 18.61 | 18.50 | | Zero-shot | Qwen 2.5 72B | 18.00 | 17.86 | | Zero-shot | GLiDRE (Ours) | 17.32 | 16.41 | | Zero-shot | Llama 3.3 70B | 15.81 | 15.71 | | Zero-shot | GPT-3.5 Turbo | 6.68 | \- |  
ในสภาวะข้อมูลกำกับมีจำกัด (Low-Resource, N=10) GLiDRE ทำคะแนน F1-Score ได้ถึง 41.73% ซึ่งสูงกว่าโมเดลฐานอย่าง ATLOP (29.48%) และ DREEAM (27.07%) อย่างมีนัยสำคัญ และในการประเมินแบบ Zero-shot GLiDRE (ขนาดเล็ก) สามารถทำคะแนน Test F1 ที่ 17.32% ซึ่งเหนือกว่า Llama 3.3 70B (15.81%)

### **เกณฑ์มาตรฐานการเชื่อมโยงความสัมพันธ์หลายช่วง (CLUTRR Kinship Benchmark)**

งานวิจัย [Neuro-symbolic Agentic SLMs (arXiv:2607.14149)](https://arxiv.org/abs/2607.14149) ได้ทำการประเมินโมเดล Gemma 3 (1B, 4B) และ Llama 3.2 (3B) บนเกณฑ์มาตรฐาน CLUTRR ซึ่งเน้นการวิเคราะห์ความสัมพันธ์ลำดับญาติที่ต้องใช้วิธีการนิรนัยหลายช่วง (Multi-hop Reasoning)  
การทดสอบชี้ให้เห็นว่าระบบ Neuro-symbolic ที่เชื่อมต่อ SLM เข้ากับเครื่องมือ extract\_facts (ทำหน้าที่สกัด triplet) และ get\_hint (ดึงคำแนะนำจาก Relational Graph Convolutional Network: RGCN) สามารถเพิ่มประสิทธิภาพการทำความเข้าใจได้ 1.5 ถึง 2 เท่าเมื่อเทียบกับโมเดลพื้นฐาน อย่างไรก็ตาม ระบบยังคงพบปัญหาขวดคอที่เรียกว่า "ความเปราะบางของการนิรนัยเชิงลำดับ" (Sequential Deductive Fragility) กล่าวคือ ข้อผิดพลาดเพียงเล็กน้อยในการสกัด triplet ตั้งแต่ประโยคแรกๆ จะถูกส่งผ่านและสะสมความล้มเหลวไปตลอดห่วงโซ่การเหตุผลเชิงตรรกะ

## **มิติการประเมินด้านประสิทธิภาพ พลังงาน และความยั่งยืน: SLM-Bench**

การประเมินความคุ้มค่าของการใช้งาน SLM ในสภาพแวดล้อมจริงจำเป็นต้องพิจารณาตัวชี้วัดด้านทรัพยากรและสิ่งแวดล้อม งานวิจัย [SLM-Bench (arXiv:2508.15478)](https://arxiv.org/abs/2508.15478) ได้จัดทำกรอบประเมินมาตรฐานสำหรับ SLM จำนวน 15 โมเดล บน 23 ชุดข้อมูล และ 4 โครงสร้างฮาร์ดแวร์ โดยวัดผลผ่าน 11 ตัวชี้วัด ซึ่งแบ่งออกเป็น 3 มิติหลัก

| มิติการประเมิน | โมเดลที่มีประสิทธิภาพสูงสุด | ลักษณะเชิงพฤติกรรมของโมเดล | ลิงก์อ้างอิงเนื้อหา |
| :---- | :---- | :---- | :---- |
| ความถูกต้อง (Correctness) | Llama-3.2-1B, Mistral-7B, Gemma-3-1B, Phi-3-3.8B | Llama-3.2-1B ได้รับจำนวนเหรียญทองด้านความถูกต้องสูงสุด รักษาสภาพเอาต์พุตได้ตรงตามโครงสร้าง | [SLM-Bench (arXiv:2508.15478)](https://arxiv.org/abs/2508.15478) |
| ประสิทธิภาพการคำนวณ (Computational Efficiency) | GPT-Neo-1.3B, TinyLlama-1.1B, ShearedLlama-2.7B | GPT-Neo-1.3B ให้ความเร็วในการประมวลผลสูงสุด (Inference Speed) และมี Latency ต่ำที่สุด | [SLM-Bench (arXiv:2508.15478)](https://arxiv.org/abs/2508.15478) |
| การใช้ทรัพยากรและสิ่งแวดล้อม (Resource Consumption) | Phi-1.5B, StableLM-3B, ShearedLlama-2.7B | Phi-1.5B ใช้พลังงานไฟฟ้าต่ำที่สุดและปล่อยก๊าซคาร์บอนไดโอไซด์น้อยที่สุดในการประมวลผล | [SLM-Bench (arXiv:2508.15478)](https://arxiv.org/abs/2508.15478) |

ข้อมูลจาก SLM-Bench แสดงให้เห็นข้อตกลงแลกเปลี่ยน (Trade-off) ที่ชัดเจน กล่าวคือ โมเดลที่มีความแม่นยำสูงอย่าง Llama-3.2-1B อาจไม่ใช่โมเดลที่ประมวลผลเร็วที่สุด ในขณะที่ Phi-1.5B เป็นตัวเลือกที่เหมาะสมที่สุดสำหรับระบบที่ต้องการความยั่งยืนและการประหยัดพลังงานขั้นสูง

## **การวิเคราะห์แนวโน้ม ทิศทางอนาคต และข้อเสนอแนะเชิงประยุกต์**

ผลจากการสังเคราะห์งานวิจัยและเกณฑ์มาตรฐานต่างๆ นำไปสู่ข้อสรุปเชิงวิศวกรรมและทิศทางการพัฒนาโมเดลภาษาขนาดเล็กสำหรับการสกัดความสัมพันธ์เชิงสามส่วนในอนาคต ดังนี้  
กระบวนทัศน์การสกัด triplet กำลังเปลี่ยนผ่านจากการพึ่งพาโมเดลขนาดใหญ่แบบโดดเดี่ยว ไปสู่ระบบไฮบริดที่ดึงจุดแข็งของทั้ง SLM และ LLM มาทำงานร่วมกัน โมเดลภาษาขนาดเล็กได้รับการพิสูจน์แล้วว่ามีประสิทธิภาพสูงในการทำหน้าที่เป็นตัวกรองเบื้องต้น (Evaluation Matrix Filter) หรือตัวย่อยประพจน์เชิงอะตอม (Atomization Processor) เพื่อสกัดคู่เอนทิตีที่มีความสัมพันธ์สูง ก่อนที่จะส่งต่อสารสนเทศที่ผ่านการกรองแล้วไปให้ LLM ดำเนินการระบุความสัมพันธ์ขั้นสูงผ่าน Prompting เทคนิคนี้ช่วยแก้ปัญหาเรื่องการตกหล่นของข้อมูล (Recall Loss) ใน LLM และลดค่าใช้จ่ายในการประมวลผลได้อย่างมีประสิทธิภาพ  
ในมิติของการประเมินผลและการนำไปใช้ในอุตสาหกรรม การประเมินผลได้ขยายจากการใช้ F1-Score บน Ground-Truth ไปสู่การใช้ออนโทโลจีเป็นตัวขับเคลื่อน (Ontology-Driven Proxy Metrics) เช่น Ontology Conformance (OC) เพื่อตรวจสอบว่า triplet ที่สกัดได้สอดคล้องกับออนโทโลจีของโดเมนหรือไม่ และ Faithfulness เพื่อตรวจสอบว่า triplet นั้นยึดโยงกับเนื้อหาต้นฉบับโดยปราศจากอาการประสาทหลอน (Hallucination) ระบบการประเมินไฮบริดที่ผสาน Regular Expressions ร่วมกับ LLM-as-a-judge จึงกลายเป็นมาตรฐานใหม่สำหรับงานสกัด triplet ในโดเมนเฉพาะทาง เช่น เอกสารทางการเงินและระบบการจัดการบันทึกข้อมูลเครื่องแม่ข่าย (System Logs)  
ในเชิงสถาปัตยกรรม โมเดลประเภท Bi-Encoder แบบจับคู่พื้นที่เวกเตอร์ เช่น GLiREL และ GLiDRE แสดงให้เห็นความเหนือกว่าโมเดลแบบ Autoregressive ทั้งในด้านความเร็วการประมวลผล ความสามารถในการสกัดความสัมพันธ์แบบ Zero-shot และความคุ้มค่าด้านพลังงาน สำหรับการพัฒนาในอนาคต การเพิ่มกลไกเสริมความสัมพันธ์ย้อนกลับแบบสองทิศทาง (Bidirectional Inverse Triplet Augmentation) และการปรับแต่งสกีมาแบบไดนามิก จะเป็นกุญแจสำคัญในการทลายข้อจำกัดด้านความเปราะบางของการนิรนัยเชิงลำดับ ช่วยให้ SLM สามารถรองรับภารกิจการสร้างกราฟความรู้ขนาดใหญ่ได้อย่างมีเสถียรภาพและยั่งยืน

#### **ผลงานที่อ้างอิง**

1\. arXiv:2404.09593v1 \[cs.CL\] 15 Apr 2024, https://arxiv.org/pdf/2404.09593 2\. A survey on cutting-edge relation extraction techniques based on, https://arxiv.org/html/2411.18157v1 3\. Slot-based Method for Accountable Relational Triple extraction \- arXiv, https://arxiv.org/html/2504.12816v3 4\. A Comprehensive Survey on Relation Extraction: Recent Advances, https://arxiv.org/html/2306.02051v3 5\. SLM-Bench: A Comprehensive Benchmark of Small Language, https://arxiv.org/html/2508.15478v1 6\. Enhancing Small Language Models Reasoning through Knowledge, https://arxiv.org/pdf/2607.14149 7\. Evaluation of a Propositioner for Triplet Extraction \- arXiv, https://arxiv.org/html/2604.02866v1 8\. GLiREL \- Generalist Model for Zero-Shot Relation Extraction \- arXiv, https://arxiv.org/html/2501.03172v1 9\. rebel/README.md at main · Babelscape/rebel \- GitHub, https://github.com/Babelscape/rebel/blob/main/README.md 10\. Babelscape/rebel-large \- Hugging Face, https://huggingface.co/Babelscape/rebel-large 11\. Babelscape/rebel-dataset · Datasets at Hugging Face, https://huggingface.co/datasets/Babelscape/rebel-dataset 12\. Towards Large Language Models Interacting with Knowledge, https://ceur-ws.org/Vol-3853/paper7.pdf 13\. jackboyla/GLiREL: Generalist and Lightweight Model for Relation, https://github.com/jackboyla/GLiREL 14\. GLiREL \- Generalist Model for Zero-Shot Relation Extraction \- Liner, https://liner.com/review/glirel-generalist-model-for-zeroshot-relation-extraction 15\. Generalist Lightweight model for Document-level Relation Extraction, https://arxiv.org/html/2508.00757v1 16\. GLiDRE: Generalist Lightweight model for Document-level Relation, https://www.researchgate.net/publication/394262622\_GLiDRE\_Generalist\_Lightweight\_model\_for\_Document-level\_Relation\_Extraction 17\. zjunlp/OneKE \- Hugging Face, https://huggingface.co/zjunlp/OneKE 18\. GitHub \- zjunlp/DeepKE: \[EMNLP 2022\] An Open Toolkit for, https://github.com/zjunlp/DeepKE 19\. arXiv:2312.01954v1 \[cs.CL\] 4 Dec 2023, https://arxiv.org/pdf/2312.01954 20\. Zero- and Few-Shots Knowledge Graph Triplet Extraction with Large, https://aclanthology.org/2024.kallm-1.2.pdf 21\. LLM-based Triplet Extraction from Financial Reports \- arXiv, https://arxiv.org/html/2602.11886v1 22\. (PDF) LLM-based Triplet Extraction from Financial Reports, https://www.researchgate.net/publication/400742860\_LLM-based\_Triplet\_Extraction\_from\_Financial\_Reports 23\. Benchmarking Small Language Models and Small Reasoning, https://arxiv.org/html/2601.07790v1 24\. Combining language models for knowledge extraction from Italian, https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1472512/full