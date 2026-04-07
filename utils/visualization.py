"""
追踪结果可视化工具
"""

import cv2
import numpy as np
from typing import Tuple, List


class TrackingVisualizer:
    """追踪结果可视化"""
    
    @staticmethod
    def draw_bbox(image: np.ndarray, bbox: Tuple, color: Tuple = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        绘制边界框
        
        Args:
            image: 输入图像
            bbox: [x1, y1, x2, y2] 或 [cx, cy, w, h]
            color: 颜色 (BGR)
            thickness: 线宽
        
        Returns:
            绘制后的图像
        """
        image = image.copy()
        
        # 判断格式并转换
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 <= x1 or y2 <= y1:  # [cx, cy, w, h] 格式
                cx, cy, w, h = bbox
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        return image
    
    @staticmethod
    def draw_center(image: np.ndarray, center: Tuple, radius: int = 5,
                   color: Tuple = (0, 0, 255)) -> np.ndarray:
        """绘制中心点"""
        image = image.copy()
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(image, (cx, cy), radius, color, -1)
        return image
    
    @staticmethod
    def draw_trajectory(image: np.ndarray, trajectory: List[Tuple],
                       color: Tuple = (0, 255, 255)) -> np.ndarray:
        """绘制运动轨迹"""
        image = image.copy()
        
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            cv2.line(image, pt1, pt2, color, 2)
        
        return image
    
    @staticmethod
    def draw_text(image: np.ndarray, text: str, position: Tuple,
                 color: Tuple = (0, 255, 0), font_scale: float = 0.7) -> np.ndarray:
        """绘制文字"""
        image = image.copy()
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, 2)
        return image
