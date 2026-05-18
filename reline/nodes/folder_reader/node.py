from __future__ import annotations

import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal
import logging

from pepeline import read, ImgFormat, ImgColor

from reline.static import Node, NodeOptions, ImageFile
from reline.utils import detect_image_color

MODE_MAP = {'rgb': ImgColor.RGB, 'gray': ImgColor.GRAY, 'dynamic': ImgColor.DYNAMIC}

Mode = Literal['rgb', 'gray', 'dynamic']


class ImageIterator:
    def __init__(self, image_paths: list, dir_path: str, mode: ImgColor):
        self.current = 0
        self.image_paths = image_paths
        self.dir_path = dir_path
        self.end = len(image_paths)
        self.mode = mode

    def __len__(self):
        return self.end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        else:
            try:
                file_path = self.image_paths[self.current]
                commonprefix = os.path.commonprefix([self.dir_path, file_path])
                dirpath = os.path.dirname(os.path.relpath(file_path, commonprefix))
                basename, _ = os.path.splitext(os.path.basename(file_path))
                data = read(file_path, self.mode, ImgFormat.F32)
                file = ImageFile(data, basename, dirpath, detect_image_color(data).is_color)
                self.current += 1
                return file
            except Exception as e:
                logging.warning(f'image {basename} not decoded due to error: {e}')
                return None


@dataclass(frozen=True)
class FolderReaderOptions(NodeOptions):
    path: str
    recursive: Optional[bool] = False
    mode: Optional[Mode] = 'dynamic'
    skip_existing_in: str = ''


class FolderReaderNode(Node[FolderReaderOptions]):
    def __init__(self, options: FolderReaderOptions):
        super().__init__(options)
        self.mode = MODE_MAP[options.mode]
        self.dir_path = os.path.abspath(self.options.path)

    def _scandir(self, dir_path: str):
        file_paths = []

        try:
            for entry in os.scandir(dir_path):
                if entry.is_file():
                    file_paths.append(os.path.abspath(entry.path))
                elif entry.is_dir() and self.options.recursive:
                    file_paths.extend(self._scandir(os.path.abspath(entry.path)))
        except OSError as e:
            logging.error(f'Error scanning directory {dir_path}: {e}')

        return file_paths

    def _filter_existing(self, file_paths: list[str]) -> list[str]:
        if not self.options.skip_existing_in:
            return file_paths
        output_dir = Path(self.options.skip_existing_in)
        if not output_dir.exists():
            return file_paths
        existing_stems = {
            path.stem
            for path in output_dir.rglob('*')
            if path.is_file() and not path.name.startswith('.')
        }
        before = len(file_paths)
        filtered = [
            file_path
            for file_path in file_paths
            if Path(file_path).stem not in existing_stems
        ]
        skipped = before - len(filtered)
        if skipped:
            logging.info(f'Skipped {skipped} files already in {output_dir}')
        return filtered

    def process(self, _) -> List[ImageFile]:
        file_paths = self._filter_existing(self._scandir(self.dir_path))
        files = []
        basename = None
        for file_path in file_paths:
            try:
                commonprefix = os.path.commonprefix([self.dir_path, file_path])
                dirpath = os.path.dirname(os.path.relpath(file_path, commonprefix))
                basename, _ = os.path.splitext(os.path.basename(file_path))

                data = read(file_path, self.mode, ImgFormat.F32)

                file = ImageFile(data, basename, dirpath, detect_image_color(data).is_color)
                files.append(file)
            except Exception as e:
                logging.warning(f'image {basename} not decoded due to error: {e}')
                continue

        return files

    def single_process(self, _) -> ImageIterator:
        file_paths = self._filter_existing(self._scandir(self.dir_path))
        return ImageIterator(file_paths, self.dir_path, self.mode)

    def video_process(self, _):
        raise ValueError('Video scale does not support folder read')
