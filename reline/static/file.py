from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ImageFile:
    data: np.ndarray
    basename: str
    dir: Optional[str] = None
    is_color: Optional[bool] = None
    skipped_nodes: list[str] = field(default_factory=list)
