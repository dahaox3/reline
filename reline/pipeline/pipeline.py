from __future__ import annotations

from typing import List, Dict

from tqdm import tqdm

from ..nodes import INTERNAL_REGISTRY
from ..static import Node
from ..nodes.file_reader import FileReaderNode
from ..nodes.folder_reader import FolderReaderNode
from ..nodes.file_writer import FileWriterNode
from ..nodes.folder_writer import FolderWriterNode
from ..nodes.upscale import UpscaleNode


class Pipeline:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.last_result = {}

    def process(self, with_tqdm: bool = True):
        data = []
        for node in tqdm(self.nodes, desc='Node Processing', disable=not with_tqdm):
            data = node.process(data)
        return data

    def process_linear(self, with_tqdm: bool = True):
        data = []
        nodes_index = 0
        save_index = 0
        while nodes_index < len(self.nodes):
            node = self.nodes[nodes_index]
            if isinstance(node, FileReaderNode | FolderReaderNode):
                data = node.single_process(data)
                for img in tqdm(data, desc='Processing Images', disable=not with_tqdm):
                    if img is None:
                        continue
                    local_node_index = nodes_index + 1
                    for node in self.nodes[local_node_index:]:
                        img = node.single_process(img)
                        local_node_index += 1
                        if isinstance(node, FolderWriterNode | FileWriterNode):
                            self.last_result = self._collect_result(img)
                            save_index = local_node_index - 1
                            break
                        if img is None:
                            break
                nodes_index = save_index
                nodes_index += 1
            else:
                nodes_index += 1
        del data

    def _collect_result(self, img):
        result = {
            'detected_color': None,
            'model_used': None,
            'skipped_nodes': [],
        }
        if img is not None:
            result['detected_color'] = 'color' if img.is_color else 'gray' if img.is_color is not None else None
            result['skipped_nodes'] = img.skipped_nodes
        for node in self.nodes:
            if isinstance(node, UpscaleNode):
                detection = getattr(node, 'last_detection', None)
                if detection is not None:
                    result['detected_color'] = 'color' if detection.is_color else 'gray'
                result['model_used'] = getattr(node, 'last_model_path', None)
                break
        return result

    @classmethod
    def from_json(cls, data: Dict) -> Pipeline:
        nodes = []

        for item in data:
            node_type = item['type']
            options_data = item['options']

            node_pair = INTERNAL_REGISTRY.get(node_type)
            options = node_pair.options(**options_data)

            node = node_pair.node(options)
            nodes.append(node)

        return Pipeline(nodes)
