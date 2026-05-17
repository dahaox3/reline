from .file_reader import FileReaderNode, FileReaderOptions
from .file_writer import FileWriterNode, FileWriterOptions
from .folder_reader import FolderReaderNode, FolderReaderOptions
from .folder_writer import FolderWriterNode, FolderWriterOptions, FileFormat
from .resize import ResizeNode, ResizeOptions
from .upscale import UpscaleNode, UpscaleOptions
from .level import LevelNode, LevelOptions
from .halftone import HalftoneNode, HalftoneOptions
from .sharp import SharpNode, SharpOptions
from .cvt_color import CvtColorNode, CvtColorOptions
from .snapshot_writer import SnapshotWriterNode, SnapshotWriterOptions
from .api_output import ApiOutputNode, ApiOutputOptions
from .registry import Registry

INTERNAL_REGISTRY = (
    Registry()
    .set('resize', ResizeNode, ResizeOptions)
    .set('file_reader', FileReaderNode, FileReaderOptions)
    .set('file_writer', FileWriterNode, FileWriterOptions)
    .set('upscale', UpscaleNode, UpscaleOptions)
    .set('folder_reader', FolderReaderNode, FolderReaderOptions)
    .set('folder_writer', FolderWriterNode, FolderWriterOptions)
    .set('level', LevelNode, LevelOptions)
    .set('halftone', HalftoneNode, HalftoneOptions)
    .set('sharp', SharpNode, SharpOptions)
    .set('cvt_color', CvtColorNode, CvtColorOptions)
    .set('snapshot_writer', SnapshotWriterNode, SnapshotWriterOptions)
    .set('api_output', ApiOutputNode, ApiOutputOptions)
)

__all__ = [
    'FileReaderNode',
    'FileWriterNode',
    'FileReaderOptions',
    'FileWriterOptions',
    'FolderReaderNode',
    'FolderReaderOptions',
    'FolderWriterNode',
    'FolderWriterOptions',
    'FileFormat',
    'ResizeNode',
    'ResizeOptions',
    'UpscaleNode',
    'UpscaleOptions',
    'LevelNode',
    'LevelOptions',
    'HalftoneNode',
    'HalftoneOptions',
    'SharpNode',
    'SharpOptions',
    'CvtColorNode',
    'CvtColorOptions',
    'SnapshotWriterNode',
    'SnapshotWriterOptions',
    'ApiOutputNode',
    'ApiOutputOptions',
    'INTERNAL_REGISTRY',
]
