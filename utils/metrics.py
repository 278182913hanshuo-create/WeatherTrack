"""
追踪性能评估指标
"""

import torch
import numpy as np
from typing import Tuple, List


class TrackingMetrics:
    """追踪性能评估指标"""
    
    @staticmethod
    def compute_iou(pred_bbox: torch.Tensor, target_bbox: torch.Tensor) -> torch.Tensor:
        """
        计算 IoU (Intersection over Union)
        
        Args:
            pred_bbox: [B, 4] (cx, cy, w, h)
            target_bbox: [B, 4] (cx, cy, w, h)
        
        Returns:
            iou: [B]
        """
        # 转换为 [x1, y1, x2, y2]
        pred_box = TrackingMetrics._convert_format(pred_bbox)
        target_box = TrackingMetrics._convert_format(target_bbox)
        
        # 计算交集
        inter_x1 = torch.max(pred_box[:, 0], target_box[:, 0])
        inter_y1 = torch.max(pred_box[:, 1], target_box[:, 1])
        inter_x2 = torch.min(pred_box[:, 2], target_box[:, 2])
        inter_y2 = torch.min(pred_box[:, 3], target_box[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        pred_area = pred_bbox[:, 2] * pred_bbox[:, 3]
        target_area = target_bbox[:, 2] * target_bbox[:, 3]
        union_area = pred_area + target_area - inter_area
        
        # 计算 IoU
        iou = inter_area / (union_area + 1e-8)
        
        return iou
    
    @staticmethod
    def compute_success_rate(ious: np.ndarray, thresholds: List[float] = [0.5, 0.75]) -> dict:
        """计算成功率"""
        results = {}
        for threshold in thresholds:
            success_count = np.sum(ious >= threshold)
            success_rate = success_count / len(ious)
            results[f'sr_{threshold}'] = success_rate
        
        return results
    
    @staticmethod
    def compute_precision(pred_centers: np.ndarray, target_centers: np.ndarray, 
                         threshold: float = 20.0) -> float:
        """计算精度"""
        distances = np.linalg.norm(pred_centers - target_centers, axis=1)
        precision = np.sum(distances <= threshold) / len(distances)
        
        return precision
    
    @staticmethod
    def _convert_format(bbox: torch.Tensor) -> torch.Tensor:
        """将 [cx, cy, w, h] 转为 [x1, y1, x2, y2]"""
        x1 = bbox[:, 0] - bbox[:, 2] / 2
        y1 = bbox[:, 1] - bbox[:, 3] / 2
        x2 = bbox[:, 0] + bbox[:, 2] / 2
        y2 = bbox[:, 1] + bbox[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)


class AverageMeter:
    """计算平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
