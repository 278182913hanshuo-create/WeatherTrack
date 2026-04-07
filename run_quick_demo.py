"""
快速演示脚本 - 无需 Weather-KITTI 数据集
在 Windows + PyCharm 上快速测试模型
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.siamese_transformer import WeatherTrackerSiamTransformer
from losses.tracking_loss import WeatherTrackingLoss


def test_model():
    """测试模型前向传播"""
    print("\n" + "=" * 70)
    print("🔥 快速模型测试 (Quick Model Test)")
    print("=" * 70)
    
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ 设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # 1. 初始化模型
        print("\n[1/5] 初始化模型...")
        model = WeatherTrackerSiamTransformer(hidden_dim=256, num_weather_types=4)
        model = model.to(device)
        print("      ✓ 模型初始化成功")
        
        # 打印模型大小
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"      参数数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 2. 生成虚拟数据
        print("\n[2/5] 生成虚拟数据...")
        batch_size = 4
        template = torch.randn(batch_size, 3, 127, 127).to(device)
        search = torch.randn(batch_size, 3, 255, 255).to(device)
        weather_encoding = torch.randn(batch_size, 4).to(device)
        
        target_bbox = torch.rand(batch_size, 4).to(device)
        target_conf = torch.ones(batch_size, 1).to(device)
        
        print(f"      ✓ Template shape: {template.shape}")
        print(f"      ✓ Search shape: {search.shape}")
        
        # 3. 前向传播
        print("\n[3/5] 前向传播...")
        model.train()
        pred_bbox, pred_conf, sim_map, weather_logits = model(
            template, search, weather_encoding
        )
        print(f"      ✓ Pred bbox shape: {pred_bbox.shape}")
        print(f"      ✓ Pred conf shape: {pred_conf.shape}")
        
        # 4. 计算损失
        print("\n[4/5] 计算损失...")
        config = {
            'bbox_weight': 1.0,
            'conf_weight': 0.5,
            'sim_weight': 0.3,
            'weather_entropy_weight': -0.1,
            'l1_loss': True
        }
        criterion = WeatherTrackingLoss(config)
        loss, loss_dict = criterion(
            pred_bbox, pred_conf, target_bbox, target_conf, sim_map, weather_logits
        )
        
        print(f"      ✓ Total Loss: {loss.item():.4f}")
        for key, val in loss_dict.items():
            print(f"        - {key}: {val:.6f}")
        
        # 5. 反向传播
        print("\n[5/5] 反向传播...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print("      ✓ 反向传播成功")
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过! 模型可以正常运行")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🌨️  WeatherTrack 目标追踪 - 快速测试".center(68) + "║")
    print("║" + "  Windows + PyCharm 运行指南".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 测试 1: 模型
    success = test_model()
    
    # 总结
    print("\n" + "=" * 70)
    print("📋 测试总结 (Summary)")
    print("=" * 70)
    print(f"✓ 模型测试: {'通过 ✅' if success else '失败 ❌'}")
    
    if success:
        print("\n🎉 所有测试通过! 你可以开始训练了")
        print("\n后续步骤:")
        print("  1. 准备自定义数据集")
        print("  2. 运行: python train_custom_dataset.py --config config/config_custom.yaml")
        print("  3. 运行: python inference_custom.py --model checkpoints/best_model.pth --video test_video.mp4")
    
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
