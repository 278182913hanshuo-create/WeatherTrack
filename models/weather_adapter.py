"""
天气自适应模块
根据天气条件动态调整特征权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherAdapterModule(nn.Module):
    """根据天气动态调整特征权重"""
    
    def __init__(self, hidden_dim, num_weather_types=4):
        super().__init__()
        
        # 天气条件下的特征适应
        self.rain_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.snow_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.fog_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.clear_adapter = nn.Linear(hidden_dim, hidden_dim)
        
        # 自适应权重生成
        self.weight_generator = nn.Sequential(
            nn.Linear(num_weather_types, 128),
            nn.ReLU(),
            nn.Linear(128, num_weather_types),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feat, weather_logits):
        """
        Args:
            feat: [B, hidden_dim]
            weather_logits: [B, 4]
        
        Returns:
            adapted_feat: [B, hidden_dim]
        """
        # 获取天气权重
        weights = self.weight_generator(weather_logits)  # [B, 4]
        
        # 应用对应的 adapter
        adapters = [self.rain_adapter, self.snow_adapter, self.fog_adapter, self.clear_adapter]
        adapted_feats = []
        for adapter in adapters:
            adapted_feats.append(adapter(feat))
        
        adapted_feats = torch.stack(adapted_feats, dim=1)  # [B, 4, hidden_dim]
        weights = weights.unsqueeze(2)  # [B, 4, 1]
        
        # 加权融合
        output = torch.sum(weights * adapted_feats, dim=1)
        
        return output
