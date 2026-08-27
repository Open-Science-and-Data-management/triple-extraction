"""รัน extract_relations กับ 3 paragraph ตัวอย่าง แล้วบันทึกผลเป็น JSON."""

import json
from relation_extractor import NER_OVERRIDES, extract_relations

# แก้ให้ spaCy แท็ก SpaceX เป็น ORG (ถูกต้อง) แทน PERSON
NER_OVERRIDES["SpaceX"] = "ORG"

SAMPLES = [
    "SpaceX was founded by Elon Musk in 2002. The company is headquartered in Hawthorne, California.",
    "Barack Obama is married to Michelle Obama, and their daughter Malia Obama was born in 1998.",
    "Instagram was acquired by Facebook in 2012. Today Instagram is a subsidiary of Meta Platforms.",
]

results = []
for text in SAMPLES:
    print("TEXT:", text)
    r = extract_relations(text)
    results.append(r)
    for rel in r["relations"]:
        print(f"  {' '.join(rel['head_text'])} --{rel['label']}--> {' '.join(rel['tail_text'])}  (score={rel['score']:.3f})")
    print()

OUT = "relation_results.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"บันทึกผลลงไฟล์: {OUT}")
