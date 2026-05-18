from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from pepeline import save


def atomic_save(data: np.ndarray, target_path: str) -> None:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=target.suffix,
        prefix=f'.{target.stem}_',
        dir=str(target.parent),
    )
    try:
        os.close(fd)
        save(data, tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
