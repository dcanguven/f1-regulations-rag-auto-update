from __future__ import annotations

import os
import time
from pathlib import Path

from app.config import settings


def _marker() -> Path:
    return Path(settings.data_dir) / ".index_generation"


def read_generation() -> str:
    try:
        return _marker().read_text().strip()
    except OSError:
        return ""


def bump_generation() -> str:
    marker = _marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    value = f"{time.time_ns()}"
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(value)
    os.replace(tmp, marker)
    return value
