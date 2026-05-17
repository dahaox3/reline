from __future__ import annotations

import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
from pepeline import save

from reline.static import ImageFile, Node, NodeOptions

FileFormat = Literal['png', 'jpeg']


@dataclass(frozen=True)
class SnapshotWriterOptions(NodeOptions):
    path: str
    format: FileFormat = 'png'


class SnapshotWriterNode(Node[SnapshotWriterOptions]):
    def __init__(self, options: SnapshotWriterOptions):
        super().__init__(options)

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        return file

    def api_process(self, file: ImageFile) -> ImageFile:
        full_path = Path(os.path.abspath(self.options.path)) / file.dir / f'{file.basename}.{self.options.format}'
        full_path.parent.mkdir(parents=True, exist_ok=True)
        target = self._unique_path(full_path)
        save(file.data, str(target))
        return file

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        counter = 1
        while True:
            target = path.parent / f'{path.stem}_{counter}{path.suffix}'
            if not target.exists():
                return target
            counter += 1

    def video_process(self, file: np.ndarray) -> np.ndarray:
        return file
