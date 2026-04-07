"""
WeatherTrack 数据处理模块
支持多种数据格式和自定义数据集
"""

from .augmentation import WeatherAugmentation, WeatherDenoiser
from .formats import BBox, ObjectAnnotation, FrameAnnotation
from .custom_dataset import CustomTrackingDataset, CustomDatasetBuilder
from .custom_dataloader import CustomDataLoader

__all__ = [
    'WeatherAugmentation',
    'WeatherDenoiser',
    'BBox',
    'ObjectAnnotation',
    'FrameAnnotation',
    'CustomTrackingDataset',
    'CustomDatasetBuilder',
    'CustomDataLoader',
]
