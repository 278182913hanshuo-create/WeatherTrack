"""
追踪系统的综合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherTrackingLoss(nn.Module):
    """综合追踪损失函数"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        self.bbox_weight = config.get('bbox_weight', 1.0)
        self.conf_weight = config.get('conf_weight', 0.5)
        self.sim_weight = config.get('sim_weight', 0.3)
        self.weather_entropy_weight = config.get('weather_entropy_weight', -0.1)
        self.use_l1 = config.get('l1_loss', True)
        
        # 损失函数
        if self.use_l1:
            self.bbox_loss_fn = nn.L1Loss()
        else:
            self.bbox_loss_fn = nn.SmoothL1Loss()
        
        self.conf_loss_fn = nn.BCEWithLogitsLoss()
        self.entropy_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred_bbox, pred_conf, target_bbox, target_conf, 
                sim_map, weather_logits, target_weather=None):
        """
        计算总损失
        
        Args:
            pred_bbox: [B, 4] 预测的 bbox (cx, cy, w, h)
            pred_conf: [B, 1] 预测的置信度
            target_bbox: [B, 4] 目标 bbox
            target_conf: [B, 1] 目标置信度
            sim_map: [B, 1, H, W] 相似度图
            weather_logits: [B, 4] 天气分类 logits
            target_weather: [B] 目标天气标签
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        batch_size = pred_bbox.size(0)
        
        # 1. BBox 回归损失
        bbox_loss = self.bbox_loss_fn(pred_bbox, target_bbox)
        
        # 2. 置信度损失
        conf_loss = self.conf_loss_fn(pred_conf, target_conf)
        
        # 3. 相似度图损失
        max_sim = torch.max(sim_map.view(batch_size, -1), dim=1)[0]
        sim_loss = -torch.log(torch.clamp(max_sim, min=1e-8)).mean()
        
        # 4. 天气分类损失
        weather_loss = torch.tensor(0.0, device=pred_bbox.device)
        if target_weather is not None:
            weather_loss = self.entropy_loss_fn(weather_logits, target_weather)
        
        # 5. 天气熵正则化
        weather_probs = F.softmax(weather_logits, dim=1)
        weather_entropy = -torch.sum(
            weather_probs * torch.log(weather_probs + 1e-8),
            dim=1
        ).mean()
        
        # 组合损失
        total_loss = (
            self.bbox_weight * bbox_loss +
            self.conf_weight * conf_loss +
            self.sim_weight * sim_loss +
            0.1 * weather_loss +
            self.weather_entropy_weight * weather_entropy
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'bbox': bbox_loss.item(),
            'conf': conf_loss.item(),
            'sim': sim_loss.item(),
            'weather': weather_loss.item() if isinstance(weather_loss, torch.Tensor) else weather_loss,
            'entropy': weather_entropy.item()
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """Focal Loss - 处理类不平衡"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        probs = torch.sigmoid(pred)
        
        focal_weight = torch.where(
            target == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma
        )
        
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()


class DIoULoss(nn.Module):
    """Distance IoU Loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_bbox, target_bbox):
        """
        Args:
            pred_bbox: [B, 4] (cx, cy, w, h)
            target_bbox: [B, 4] (cx, cy, w, h)
        """
        # 转换为 [x1, y1, x2, y2]
        pred_box = self._convert_format(pred_bbox)
        target_box = self._convert_format(target_bbox)
        
        # 计算 IoU
        inter_x1 = torch.max(pred_box[:, 0], target_box[:, 0])
        inter_y1 = torch.max(pred_box[:, 1], target_box[:, 1])
        inter_x2 = torch.min(pred_box[:, 2], target_box[:, 2])
        inter_y2 = torch.min(pred_box[:, 3], target_box[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = pred_bbox[:, 2] * pred_bbox[:, 3]
        target_area = target_bbox[:, 2] * target_bbox[:, 3]
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-8)
        
        # 计算中心距离
        pred_center = pred_bbox[:, :2]
        target_center = target_bbox[:, :2]
        center_dist = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # 外接框对角线
        enclose_x1 = torch.min(pred_box[:, 0], target_box[:, 0])
        enclose_y1 = torch.min(pred_box[:, 1], target_box[:, 1])
        enclose_x2 = torch.max(pred_box[:, 2], target_box[:, 2])
        enclose_y2 = torch.max(pred_box[:, 3], target_box[:, 3])
        
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # DIoU
        diou = iou - center_dist / (enclose_diag + 1e-8)
        
        return (1 - diou).mean()
    
    @staticmethod
    def _convert_format(bbox):
        """将 [cx, cy, w, h] 转为 [x1, y1, x2, y2]"""
        x1 = bbox[:, 0] - bbox[:, 2] / 2
        y1 = bbox[:, 1] - bbox[:, 3] / 2
        x2 = bbox[:, 0] + bbox[:, 2] / 2
        y2 = bbox[:, 1] + bbox[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
