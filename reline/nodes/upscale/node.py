from dataclasses import dataclass
from typing import Callable, Optional, List, Literal

import numpy as np
import torch.cuda
from resselt import load_from_file
from resr.tiling import MaxTileSize, ExactTileSize, NoTiling, process_tiles
from pepeline import cvt_color, CVTColor
from reline.static import Node, NodeOptions, ImageFile
from reline.utils import ColorDetectionResult, detect_image_color
import logging

Tiler = Literal['exact', 'max', 'no_tiling']
DType = Literal['F32', 'F16', 'BF16']
ColorDetectMode = Literal['auto', 'force_color', 'force_gray']
ModelCacheMode = Literal['low_memory', 'high_memory']
ModelSelector = Callable[[ImageFile, bool], Optional[str]]


def empty_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


@dataclass(frozen=True)
class UpscaleOptions(NodeOptions):
    model: str
    tiler: Tiler
    target_scale: Optional[int] = None
    dtype: Optional[DType] = 'F32'
    exact_tiler_size: Optional[int] = 256
    allow_cpu_upscale: Optional[bool] = False
    auto_detect_color: Optional[bool] = False
    color_model: Optional[str] = None
    gray_model: Optional[str] = None
    color_detect_mode: Optional[ColorDetectMode] = 'auto'
    model_cache_mode: Optional[ModelCacheMode] = 'low_memory'


class UpscaleNode(Node[UpscaleOptions]):
    def __init__(self, options: UpscaleOptions):
        super().__init__(options)

        if not torch.cuda.is_available() and not options.allow_cpu_upscale:
            raise BaseException('CUDA is not available. If you want scale with CPU use `allow_cpu_upscale` option')
        self.target_scale = options.target_scale
        self.model = None
        self.model_path = None
        self.model_cache = {}
        self.last_detection = None
        self.last_model_path = None
        self.model_selector: Optional[ModelSelector] = None
        self.tiler = self._create_tiler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if options.dtype == 'F16':
            self.dtype = torch.half
        elif options.dtype == 'BF16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        if not options.auto_detect_color or options.model:
            self._switch_model(options.model)
        if self.device == 'cuda':
            empty_cuda_cache()

    def set_model_selector(self, selector: Optional[ModelSelector]):
        self.model_selector = selector

    def _switch_model(self, model_path: Optional[str]):
        if not model_path:
            raise ValueError('Upscale model path is empty')
        if self.model_path == model_path and self.model is not None:
            return
        if self.options.model_cache_mode == 'high_memory' and model_path in self.model_cache:
            self.model = self.model_cache[model_path]
            self.model_path = model_path
            return
        if self.model is not None:
            if self.options.model_cache_mode == 'high_memory':
                self.model_cache[self.model_path] = self.model
            else:
                del self.model
            self.model = None
            if self.device == 'cuda' and self.options.model_cache_mode != 'high_memory':
                empty_cuda_cache()
        self.model = load_from_file(model_path)
        self.model_path = model_path
        if self.options.model_cache_mode == 'high_memory':
            self.model_cache[model_path] = self.model

    def _image_label(self, file: ImageFile) -> str:
        return f'{file.dir}/{file.basename}' if file.dir else file.basename

    def _select_model(self, file: ImageFile, detection: ColorDetectionResult) -> Optional[str]:
        if self.model_selector is not None:
            return self.model_selector(file, detection.is_color)
        if not self.options.auto_detect_color:
            return self.options.model

        preferred_model = self.options.color_model if detection.is_color else self.options.gray_model
        fallback_model = self.options.model
        image_type = 'color' if detection.is_color else 'gray'
        if preferred_model:
            return preferred_model
        if fallback_model:
            logging.warning('%s image `%s` has no dedicated %s model configured; using fallback model `%s`', image_type, self._image_label(file), image_type, fallback_model)
            return fallback_model
        return None

    def _img_ch_to_model_ch(self, img: np.ndarray) -> np.ndarray:
        img_shape = img.shape
        img = img.squeeze()
        if self.model is None:
            raise ValueError('Upscale model is not loaded')
        if self.model.parameters_info.in_channels == 3:
            if len(img_shape) == 2:
                img = cvt_color(img, CVTColor.Gray2RGB)
        elif self.model.parameters_info.in_channels == 1:
            if len(img_shape) == 3:
                img = cvt_color(img.squeeze(), CVTColor.RGB2Gray_2020)
        else:
            logging.error('model format is not currently supported')
        return img

    def _process_image(self, file: ImageFile) -> Optional[ImageFile]:
        if self.options.auto_detect_color:
            if file.is_color is None or self.options.color_detect_mode in ('force_color', 'force_gray'):
                detection = detect_image_color(file.data, self.options.color_detect_mode)
                file.is_color = detection.is_color
            else:
                detection = ColorDetectionResult(file.is_color, reason='metadata')
        else:
            detection = ColorDetectionResult(file.is_color or False, reason='disabled')
        model_path = self._select_model(file, detection)
        label = self._image_label(file)
        metrics = ''
        if detection.saturated_ratio is not None and detection.rgb_diff_mean is not None:
            metrics = f', saturated_ratio={detection.saturated_ratio:.4f}, rgb_diff_mean={detection.rgb_diff_mean:.2f}'
        if not model_path:
            logging.error('No valid upscale model for `%s` (%s%s); skipping image', label, 'color' if detection.is_color else 'gray', metrics)
            return None

        try:
            self._switch_model(model_path)
        except Exception as e:
            logging.error('Failed to load upscale model `%s` for `%s`: %s; skipping image', model_path, label, e)
            return None
        self.last_detection = detection
        self.last_model_path = model_path
        logging.info('Upscale `%s`: detected=%s, reason=%s%s, model=%s', label, 'color' if detection.is_color else 'gray', detection.reason, metrics, model_path)
        img = self._img_ch_to_model_ch(file.data)
        file.data = process_tiles(
            img,
            tiler=self.tiler,
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            model_scale=self.model.parameters_info.upscale,
            target_scale=self.target_scale,
            channels=self.model.parameters_info.in_channels,
        )
        return file

    def _create_tiler(self):
        match self.options.tiler:
            case 'exact':
                if self.options.exact_tiler_size is None:
                    raise ValueError('Exact tiler requires `exact_tiler_size` param')
                return ExactTileSize(self.options.exact_tiler_size)
            case 'max':
                return MaxTileSize()
            case 'no_tiling':
                return NoTiling()
            case _:
                raise ValueError(f'Unknown tiler option `{self.options.tiler}`')

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        processed_files = []
        for file in files:
            processed = self._process_image(file)
            if processed is not None:
                processed_files.append(processed)
        return processed_files

    def single_process(self, file: ImageFile) -> Optional[ImageFile]:
        return self._process_image(file)

    def video_process(self, file: np.ndarray) -> np.ndarray:
        if self.model is None:
            self._switch_model(self.options.model)
        img = self._img_ch_to_model_ch(file)
        file = process_tiles(
            img,
            tiler=self.tiler,
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            model_scale=self.model.parameters_info.upscale,
            target_scale=self.target_scale,
            channels=self.model.parameters_info.in_channels,
        )
        return file
