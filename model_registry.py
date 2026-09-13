"""Atomic model promotion with a verifiable, reversible local release history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np

from settings import MODEL_PATH


def publish(model, path: Path, probe) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    backup = path.parent / "backups" / "models" / uuid4().hex
    backup.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, temporary, compress=3)
        loaded = joblib.load(temporary)
        actual, expected = loaded.predict_proba(probe), model.predict_proba(probe)
        if not np.isfinite(actual).all() or not np.allclose(actual, expected):
            raise ValueError("Model serialization verification failed")
        if path.exists():
            shutil.copy2(path, backup / path.name)
            old_report = path.with_suffix(".metrics.json")
            if old_report.exists():
                shutil.copy2(old_report, backup / old_report.name)
        report = path.with_suffix(".metrics.json")
        report_tmp = report.with_suffix(f".{uuid4().hex}.tmp")
        report_tmp.write_text(
            json.dumps(model.ufc_metadata_, indent=2, allow_nan=False), encoding="utf-8"
        )
        # The model embeds its own authoritative report; a sidecar is never used for predictions.
        os.replace(temporary, path)
        os.replace(report_tmp, report)
        result = {
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "previous_model_directory": str(backup),
        }
        (backup / "promotion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        temporary.unlink(missing_ok=True)


def restore(backup: Path, path: Path = MODEL_PATH):
    source = Path(backup) / Path(path).name
    if not source.is_file():
        raise ValueError("No saved model in that backup directory")
    model = joblib.load(source)
    if not hasattr(model, "ufc_metadata_") or not hasattr(model, "predict_proba"):
        raise ValueError("Backup is not a recognized local model")
    if Path(path).exists():
        retained = Path(path).parent / "backups" / "models" / uuid4().hex
        retained.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, retained / Path(path).name)
        old_report = Path(path).with_suffix(".metrics.json")
        if old_report.exists():
            shutil.copy2(old_report, retained / old_report.name)
    temporary = Path(path).with_suffix(f".{uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, path)
    Path(path).with_suffix(".metrics.json").write_text(
        json.dumps(model.ufc_metadata_, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore a trusted local model backup")
    parser.add_argument("backup", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    restore(args.backup, args.model)
