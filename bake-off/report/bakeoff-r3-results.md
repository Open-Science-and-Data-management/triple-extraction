# Bake-off r3 — ขยายแผนที่สู่ code/instruction-LLM dedicated IE

คุมตัวแปรกับ r2 ทุกอย่างยกเว้นตัวโมเดล (ประโยค/seed schema/GPU เดิม) — ตัวใหม่ 2 ตัวเป็น decoder 7B Q4 nf4

## แผนที่ precision × ms (7 จุด)

| model | paradigm | precision ~ (best th) | ms/ประโยค | VRAM peak | schema |
|---|---|---|---|---|---|
| gliner-relex | encoder | ~0.88 @ th 0.9 | 15.5 | 1.29 GiB | seed schema |
| glirel | encoder | ~0.14 @ th 0.25 | 31.2 | 3.32 GiB | seed schema |
| gliner-pyrheads | encoder | ~0.53 @ th 0.85 | 46.7 | 2.09 GiB | seed schema |
| relik | encoder | ~0.00 @ th 0.5 | 151.0 | 1.79 GiB | ปิด (native NYT) |
| nuextract | decoder | ~1.00 @ th 0.98 | 976.6 | 4.39 GiB | seed schema |

```
precision  (ms 15–977, log scale)
1.00 │                                               D
0.95 │                                                
0.90 │E                                               
0.85 │                                                
0.80 │                                                
0.75 │                                                
0.70 │                                                
0.65 │                                                
0.60 │                                                
0.55 │             E                                  
0.50 │                                                
0.45 │                                                
0.40 │                                                
0.35 │                                                
0.30 │                                                
0.25 │                                                
0.20 │                                                
0.15 │        E                                       
0.10 │                                                
0.05 │                                                
0.00 │                          E                     
     └───────────────────────────────────────────────
      E = encoder   D = decoder   * = จุดชนกัน
```

precision = rate ด้วยตาทุก unique triple ครั้งเดียวแล้ว slice ทุก threshold จาก raw scores เดียว

## ตัดแล้วและเหตุผล

- **InstructUIE** — T5-11B fp32 ~45GB — Q4 ≈ 6GB+ ไม่มี headroom (การ์ดใช้เป็นจอ)
- **CodeKGC** — ไม่มี public checkpoint (โค้ดใน zjunlp/DeepKE ต้อง fine-tune เอง) — Never: fine-tune CodeLlama
- **GoLLIE-13B** — VRAM 6GB — เหตุผลเดียวกับ InstructUIE
- **knowcoder** — ตายตอน smoke: checkpoint golaxy/KnowCoder-7B-IE 13GB ดาวน์โหลดไม่เสร็จ — สาย ~3.3MB/s (45 นาที ได้ 4.7GB) — smoke ไม่เคยเริ่ม load ที่ GPU เลย — human สั่งตัด 2026-09-05 ก่อนวัดผล hf-transfer

- **ถ้า KnowCoder/GoLLIE ไม่ปรากฏในตาราง** = smoke ไม่ผ่านและบันทึกเหตุผลไว้ที่นี่ (ไม่มีตัวหายเงียบ)
