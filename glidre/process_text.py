from glidre import GLiDRE

# โหลดโมเดลครั้งเดียวตอน import (ตัวเดียวกัน reuse ได้ทั้ง module)
_model = GLiDRE.from_pretrained("cea-list-ia/glidre_large")


def process_text(
    text: str,
    labels: list[str],
    mentions: list[dict],
    threshold: float = 0.3,
    multi_label: bool = False,
) -> list[dict]:
    """Predict relations between entity mentions in a paragraph."""
    return _model.predict_entities(
        text=text,
        labels=labels,
        mentions=mentions,
        threshold=threshold,
        multi_label=multi_label,
    )


if __name__ == "__main__":
    text = (
        "The Loud Tour was the fourth overall and third world concert tour "
        "by Barbadian recording artist Rihanna."
    )
    labels = ["COUNTRY_OF_CITIZENSHIP", "PUBLICATION_DATE", "PART_OF"]
    mentions = [
        {"id": 0, "type": "LOC", "mentions": [{"value": "Barbadian", "start": 69, "end": 78}]},
        {"id": 1, "type": "PER", "mentions": [{"value": "Rihanna", "start": 96, "end": 103}]},
    ]
    relations = process_text(text, labels, mentions)
    for r in relations:
        print(r["entity_1"], "|", r["relation_type"], "|", r["entity_2"])
