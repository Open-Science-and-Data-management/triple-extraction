"""ดาวน์โหลด checkpoint GLiNER-Relex → Settings().MODEL_DIR (ครั้งเดียว, idempotent)

รัน: uv run python -m extraction_api.download_model
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from extraction_api.settings import Settings


def is_downloaded(model_dir) -> bool:
    # repo นี้ใช้ gliner_config.json (ไม่มี config.json)
    has_config = any((model_dir / f).exists() for f in ("config.json", "gliner_config.json"))
    has_weights = any((model_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
    return has_config and has_weights


def main() -> int:
    settings = Settings()
    if is_downloaded(settings.model_dir):
        print(f"มี checkpoint แล้ว ข้าม: {settings.model_dir}")
        return 0
    path = snapshot_download("knowledgator/gliner-relex-multi-v1.0", local_dir=settings.model_dir)
    if not is_downloaded(settings.model_dir):
        print(f"ดาวน์โหลดแล้วแต่ไฟล์จำเป็นไม่ครบ: {path}", file=sys.stderr)
        return 1
    print(f"ดาวน์โหลดแล้ว: {settings.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
