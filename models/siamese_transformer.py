"""
WeatherTrack 核心模型：Siamese-Transformer 混合架构
"""

import torch
import torch.nn as nn
from torchvision import models

from .backbone import SiameseHead, BBoxRegressionHead, ChannelAttention
from .weather_adapter import WeatherAdapterModule


class WeatherTrackerSiamTransformer(nn.Module):
    """Weather-KITTI 目标追踪模型"""
    
    def __init__(self, hidden_dim=256, num_weather_types=4):
        super().__init__()
        
        # Backbone: ResNet50
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Siamese 头
        self.siamese_head = SiameseHead(2048, hidden_dim)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # 天气分类器
        self.weather_classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_weather_types)
        )
        
        # 天气适应器
        self.weather_adapter = WeatherAdapterModule(hidden_dim, num_weather_types)
        
        # 预测头
        self.bbox_head = BBoxRegressionHead(hidden_dim, 4)
        self.conf_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, template, search, weather_encoding):
        """
        前向传播
        
        Args:
            template: [B, 3, 127, 127] - 第一帧目标区域
            search: [B, 3, 255, 255] - 搜索区域
            weather_encoding: [B, 4] - 天气特征
        
        Returns:
            bbox: [B, 4] - 预测的 bbox (cx, cy, w, h)
            conf: [B, 1] - 预测的置信度
            sim_map: [B, 1, H, W] - 相似度图
            weather_logits: [B, 4] - 天气分类 logits
        """
        B = template.size(0)
        
        # 1. 特征提取
        z_feat = self.backbone(template)  # [B, 2048, 4, 4]
        x_feat = self.backbone(search)     # [B, 2048, 8, 8]
        
        # 2. 天气分类
        z_pool = torch.nn.functional.adaptive_avg_pool2d(z_feat, 1).view(B, -1)
        weather_logits = self.weather_classifier(z_pool)
        
        # 3. Siamese 相似度
        z_sim = self.siamese_head(z_feat)
        x_sim = self.siamese_head(x_feat)
        
        # 4. 相似度图
        sim_map = self._compute_similarity_map(z_sim, x_sim)
        
        # 5. 特征融合
        x_sim_pool = torch.nn.functional.adaptive_avg_pool2d(x_sim, 1).view(B, -1)
        sim_pool = torch.nn.functional.adaptive_avg_pool2d(sim_map, 1).view(B, -1)
        feat_combined = torch.cat([x_sim_pool, sim_pool], dim=1)
        
        # 6. Transformer 处理
        feat_combined = feat_combined.unsqueeze(1)
        feat_transformed = self.transformer(feat_combined)
        feat_transformed = feat_transformed.squeeze(1)
        
        # 7. 天气自适应
        adapted_feat = self.weather_adapter(feat_transformed, weather_encoding)
        
        # 8. 预测
        bbox = self.bbox_head(adapted_feat)
        conf = self.conf_head(adapted_feat)
        
        return bbox, conf, sim_map, weather_logits
    
    def _compute_similarity_map(self, template, search):
        """计算相似度图 (交叉相关)"""
        B, C, H1, W1 = template.shape
        _, _, H2, W2 = search.shape
        
        # 正规化
        template = template / (torch.norm(template, p=2, dim=1, keepdim=True) + 1e-8)
        search = search / (torch.norm(search, p=2, dim=1, keepdim=True) + 1e-8)
        
        # 计算相似度图
        sim_map = torch.zeros(B, 1, H2-H1+1, W2-W1+1, device=template.device)
        
        for i in range(H2-H1+1):
            for j in range(W2-W1+1):
                patch = search[:, :, i:i+H1, j:j+W1]
                sim_map[:, :, i, j] = torch.sum(template * patch, dim=[1, 2, 3], keepdim=True).squeeze()
        
        return sim_map
