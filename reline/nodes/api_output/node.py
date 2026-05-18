from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np

from reline.static import ImageFile, Node, NodeOptions

FileFormat = Literal['png', 'jpeg']


@dataclass(frozen=True)
class ApiOutputOptions(NodeOptions):
    format: FileFormat = 'jpeg'


class ApiOutputNode(Node[ApiOutputOptions]):
    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        return file
