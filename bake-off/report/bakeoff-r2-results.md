# Bake-off r2 — Encoder zero-shot dedicated IE

| model | precision ~ (best th) | ms/ประโยค | VRAM peak | schema |
|---|---|---|---|---|
| gliner-relex | ~0.88 @ th 0.9 | 15.5 | 1.29 GiB | seed schema |
| glirel | ~0.14 @ th 0.25 | 31.2 | 3.32 GiB | seed schema |
| gliner-pyrheads | ~0.53 @ th 0.85 | 46.7 | 2.09 GiB | seed schema |
| relik | ~0.00 @ th 0.5 | 151.0 | 1.79 GiB | ปิด (native NYT) |
| nuextract | ~1.00 @ th 0.98 | 976.6 | 4.39 GiB | seed schema |

precision = rate ด้วยตาทุก unique triple ครั้งเดียว (Step 3) แล้ว slice ทุก threshold จาก raw scores เดียว

## gliner-relex

233 triples (232 unique) · 15.5 ms/ประโยค (หลัง warm-up) · VRAM peak: 1.29 GiB

| th | triples | precision ~ | ประโยคว่าง |
|---|---|---|---|
| 0.3 | 233 | ~0.39 | 0 |
| 0.5 | 139 | ~0.52 | 0 |
| 0.7 | 92 | ~0.66 | 1 |
| 0.9 | 33 | ~0.88 | 9 |

**best:** th 0.9 — precision ~0.88 (33 triples, ว่าง 9/28)

**แตกตาม category @ th 0.9**

| category | triples | ประโยคว่าง/รวม |
|---|---|---|
| alias | 4 | 0/2 |
| comparison | 10 | 0/4 |
| training | 7 | 2/5 |
| benchmark | 3 | 1/3 |
| effect | 8 | 1/7 |
| multi-rel | 5 | 0/4 |
| hard | 3 | 5/7 |

## glirel

56 triples (56 unique) · 31.2 ms/ประโยค (หลัง warm-up) · VRAM peak: 3.32 GiB

| th | triples | precision ~ | ประโยคว่าง |
|---|---|---|---|
| 0.25 | 56 | ~0.14 | 15 |
| 0.3 | 56 | ~0.14 | 15 |
| 0.35 | 47 | ~0.11 | 18 |
| 0.45 | 29 | ~0.07 | 23 |

**best:** th 0.25 — precision ~0.14 (56 triples, ว่าง 15/28)

**แตกตาม category @ th 0.25**

| category | triples | ประโยคว่าง/รวม |
|---|---|---|
| alias | 4 | 1/2 |
| comparison | 18 | 0/4 |
| training | 6 | 4/5 |
| benchmark | 5 | 0/3 |
| effect | 7 | 5/7 |
| multi-rel | 9 | 2/4 |
| hard | 15 | 4/7 |

## gliner-pyrheads

216 triples (212 unique) · 46.7 ms/ประโยค (หลัง warm-up) · VRAM peak: 2.09 GiB

| th | triples | precision ~ | ประโยคว่าง |
|---|---|---|---|
| 0.4 | 209 | ~0.26 | 1 |
| 0.55 | 178 | ~0.29 | 1 |
| 0.7 | 131 | ~0.37 | 1 |
| 0.85 | 73 | ~0.53 | 5 |

**best:** th 0.85 — precision ~0.53 (73 triples, ว่าง 5/28)

**แตกตาม category @ th 0.85**

| category | triples | ประโยคว่าง/รวม |
|---|---|---|
| alias | 5 | 0/2 |
| comparison | 24 | 0/4 |
| training | 16 | 0/5 |
| benchmark | 13 | 0/3 |
| effect | 13 | 1/7 |
| multi-rel | 8 | 0/4 |
| hard | 8 | 4/7 |

## relik

33 triples (33 unique) · 151.0 ms/ประโยค (หลัง warm-up) · VRAM peak: 1.79 GiB

| th | triples | precision ~ | ประโยคว่าง |
|---|---|---|---|
| 0.5 | 33 | ~0.00 | 21 |
| 0.7 | 1 | ~0.00 | 27 |
| 0.9 | 0 | – | 28 |

**best:** th 0.5 — precision ~0.00 (33 triples, ว่าง 21/28)

**แตกตาม category @ th 0.5**

| category | triples | ประโยคว่าง/รวม |
|---|---|---|
| alias | 0 | 2/2 |
| comparison | 1 | 3/4 |
| training | 5 | 3/5 |
| benchmark | 1 | 2/3 |
| effect | 23 | 5/7 |
| multi-rel | 0 | 4/4 |
| hard | 4 | 5/7 |

## nuextract

42 triples (42 unique) · 976.6 ms/ประโยค (หลัง warm-up) · VRAM peak: 4.39 GiB

| th | triples | precision ~ | ประโยคว่าง |
|---|---|---|---|
| 0.85 | 42 | ~0.55 | 5 |
| 0.9 | 39 | ~0.56 | 5 |
| 0.94 | 27 | ~0.63 | 8 |
| 0.98 | 2 | ~1.00 | 26 |

**best:** th 0.98 — precision ~1.00 (2 triples, ว่าง 26/28)

**แตกตาม category @ th 0.98**

| category | triples | ประโยคว่าง/รวม |
|---|---|---|
| alias | 0 | 2/2 |
| comparison | 2 | 2/4 |
| training | 0 | 5/5 |
| benchmark | 1 | 2/3 |
| effect | 0 | 7/7 |
| multi-rel | 0 | 4/4 |
| hard | 0 | 7/7 |
