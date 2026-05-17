from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from reline.static import ImageFile, Node, NodeOptions


@dataclass(frozen=True)
class ApiOutputOptions(NodeOptions):
    pass


class ApiOutputNode(Node[ApiOutputOptions]):
    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        return file
