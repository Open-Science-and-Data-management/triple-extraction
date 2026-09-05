"""ค่า config ทั้งหมดอ่านจาก .env (มี default ครบ — ไม่มี .env ก็รันได้)"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    retention_days: int = 7
    max_results_mb: int = 500
    webhook_enabled: bool = True
    callback_timeout: float = 10.0
    max_files: int = 20
    max_bytes: int = 5_242_880
    default_threshold: float = 0.9
    model_dir: Path = Path("models/gliner-relex-multi-v1.0")
    db_path: Path = Path("data/jobs.db")
    results_dir: Path = Path("data/results")
    device: str = "cuda"
