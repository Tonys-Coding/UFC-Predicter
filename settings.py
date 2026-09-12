"""Shared local paths and rotating application logs. No secrets are logged."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
DB_PATH = Path(os.getenv("UFC_DB_PATH", str(ROOT / "ufc_analytics.db"))).expanduser()
MODEL_PATH = Path(os.getenv("UFC_MODEL_PATH", str(ROOT / "ufc_brain.pkl"))).expanduser()
CACHE_DIR = ROOT / "cache"
TIMEZONE = os.getenv("APP_TIMEZONE", "America/Chicago")


def configure_logging() -> None:
    logger = logging.getLogger("ufc")
    if logger.handlers:
        return
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        (ROOT / "logs").mkdir(exist_ok=True)
        handler = RotatingFileHandler(ROOT / "logs" / "ufc.log", maxBytes=2_000_000, backupCount=4)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except OSError:
        logger.warning("File logging unavailable; continuing with terminal logging.")
    logger.propagate = False
