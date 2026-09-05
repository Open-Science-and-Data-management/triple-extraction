"""ดาวน์โหลด checkpoint GLiNER-Relex → Settings().MODEL_DIR (ครั้งเดียว, idempotent)

รัน: uv run python -m extraction_api.download_model
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from extraction_api.settings import Settings

# ไฟล์ที่ from_pretrained ต้องมี — safetensors หรือ bin อย่างน้อยหนึ่ง
REQUIRED = ["config.json"]


def is_downloaded(model_dir) -> bool:
    if not (model_dir / "config.json").exists():
        return False
    return any(
        (model_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin")
    )


def main() -> int:
    settings = Settings()
    if is_downloaded(settings.model_dir):
        print(f"มี checkpoint แล้ว ข้าม: {settings.model_dir}")
        return 0
    path = snapshot_download(
        "knowledgator/gliner-relex-multi-v1.0", local_dir=settings.model_dir
    )
    if not is_downloaded(settings.model_dir):
        print(f"ดาวน์โหลดแล้วแต่ไฟล์จำเป็นไม่ครบ: {path}", file=sys.stderr)
        return 1
    print(f"ดาวน์โหลดแล้ว: {settings.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
