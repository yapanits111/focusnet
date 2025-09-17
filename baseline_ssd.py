# baseline_ssd.py
# Baseline SSD model WITHOUT CBAM attention for thesis comparison
# This is the standard SSD implementation that FocusNet will be compared against
# 
# Purpose: Demonstrate the effectiveness of adding CBAM attention to SSD
# Used for: Statistical comparison in thesis evaluation (Wilcoxon signed-rank test)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

class BaselineMobileNetV3Backbone(nn.Module):
    """
    Standard MobileNetV3 backbone WITHOUT CBAM attention
    This is the baseline for comparison with FocusNet
    """
    def __init__(self, pretrained=True, freeze_bn=False):
        super().__init__()
        
        # Load pretrained MobileNetV3-Large
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        
        # Extract backbone features (same structure as FocusNet but NO CBAM)
        self.stem = nn.Sequential(base.features[0])  # stride 2
        
        # Split stages to tap features at multiple scales
        self.stage1 = nn.Sequential(*base.features[1:6])   # stride 4/8
        self.stage2 = nn.Sequential(*base.features[6:12])  # stride 8/16
        self.stage3 = nn.Sequential(*base.features[12:])   # stride 16
        
        # Extra SSD layers for more detection scales (standard approach)
        self.extra1 = nn.Sequential(
            nn.Conv2d(self._out_channels(self.stage3), 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True)
        )
        
        self.extra2 = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )
        
        # Freeze batch norm if requested
        if freeze_bn:
            self._freeze_bn()
    
    def _out_channels(self, stage):
        """Get output channels of a stage"""
        for layer in reversed(stage):
            if hasattr(layer, 'out_channels'):
                return layer.out_channels
            elif hasattr(layer, 'block'):
                return layer.block.out_channels
        return 960  # Default for MobileNetV3-Large final stage
    
    def _freeze_bn(self):
        """Freeze batch normalization layers"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass extracting multi-scale features
        NO CBAM attention applied (this is the key difference from FocusNet)
        
        Returns:
            list: Feature maps at different scales for SSD detection
        """
        # Initial convolution
        x = self.stem(x)  # stride 2
        
        # Extract features at multiple scales
        feat1 = self.stage1(x)   # stride 4/8
        feat2 = self.stage2(feat1)  # stride 8/16  
        feat3 = self.stage3(feat2)  # stride 16
        
        # Additional feature scales
        feat4 = self.extra1(feat3)  # stride 32
        feat5 = self.extra2(feat4)  # stride 64
        
        # Return features for SSD head (NO CBAM enhancement)
        return [feat2, feat3, feat4, feat5]

class BaselineSSD(nn.Module):
    """
    Baseline SSD model using standard MobileNetV3 backbone
    This is the baseline model that FocusNet (with CBAM) will be compared against
    
    Architecture: Standard SSD + MobileNetV3 (NO CBAM)
    Purpose: Demonstrate effectiveness of CBAM attention in thesis evaluation
    """
    
    def __init__(self, num_classes, pretrained=True, freeze_backbone_bn=False):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone: Standard MobileNetV3 (NO CBAM)
        self.backbone = BaselineMobileNetV3Backbone(
            pretrained=pretrained,
            freeze_bn=freeze_backbone_bn
        )
        
        # SSD detection head (same as FocusNet for fair comparison)
        from ssd_head import SSDHead
        
        # Feature map sizes for different scales
        feature_sizes = [(40, 40), (20, 20), (10, 10), (5, 5)]  # For 320x320 input
        
        self.ssd_head = SSDHead(
            in_channels=[112, 960, 512, 256],  # MobileNetV3 channel counts
            num_classes=num_classes,
            feature_sizes=feature_sizes
        )
        
        print(f"✅ Baseline SSD created:")
        print(f"   Architecture: Standard SSD + MobileNetV3")
        print(f"   Attention: None (baseline for comparison)")
        print(f"   Classes: {num_classes}")
        print(f"   Purpose: Baseline model for thesis evaluation")
    
    def forward(self, x):
        """
        Forward pass through baseline SSD
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W]
            
        Returns:
            tuple: (class_logits, box_deltas, anchors)
                - class_logits: Classification predictions
                - box_deltas: Box regression predictions  
                - anchors: Default anchor boxes
        """
        # Extract features using standard MobileNetV3 (NO CBAM)
        features = self.backbone(x)
        
        # SSD detection head
        cls_logits, box_deltas, anchors = self.ssd_head(features)
        
        return cls_logits, box_deltas, anchors
    
    def get_architecture_info(self):
        """Return information about the model architecture"""
        return {
            'name': 'Baseline SSD',
            'backbone': 'MobileNetV3-Large', 
            'attention': None,
            'purpose': 'Baseline for comparison with FocusNet',
            'num_classes': self.num_classes
        }

def create_baseline_ssd(num_classes, pretrained=True, device='cuda'):
    """
    Create baseline SSD model for thesis comparison
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Use pretrained MobileNetV3 weights
        device (str): Device to place model on
    
    Returns:
        BaselineSSD: Baseline model for comparison
    """
    print("🔧 Creating Baseline SSD for thesis comparison...")
    
    model = BaselineSSD(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone_bn=False
    )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Baseline SSD Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Memory footprint: ~{total_params * 4 / 1e6:.1f} MB")
    print(f"   Device: {device}")
    
    return model

def compare_architectures(focusnet_model, baseline_model):
    """
    Compare FocusNet and Baseline SSD architectures
    Useful for thesis documentation
    
    Args:
        focusnet_model: FocusNet model (with CBAM)
        baseline_model: Baseline SSD model (without CBAM)
    
    Returns:
        dict: Comparison statistics
    """
    focusnet_params = sum(p.numel() for p in focusnet_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    comparison = {
        'focusnet': {
            'name': 'FocusNet (SSD + MobileNetV3 + CBAM)',
            'parameters': focusnet_params,
            'attention': 'CBAM (Channel + Spatial)',
            'innovation': 'Novel attention mechanism'
        },
        'baseline': {
            'name': 'Baseline SSD (SSD + MobileNetV3)',
            'parameters': baseline_params,  
            'attention': 'None',
            'innovation': 'Standard approach'
        },
        'difference': {
            'parameter_increase': focusnet_params - baseline_params,
            'percentage_increase': ((focusnet_params - baseline_params) / baseline_params) * 100
        }
    }
    
    print("📊 ARCHITECTURE COMPARISON FOR THESIS:")
    print("=" * 50)
    print(f"FocusNet (Proposed):")
    print(f"   Parameters: {focusnet_params:,}")
    print(f"   Attention: CBAM (Channel + Spatial)")
    print(f"   Innovation: Novel attention-enhanced SSD")
    
    print(f"\nBaseline SSD (Comparison):")
    print(f"   Parameters: {baseline_params:,}")
    print(f"   Attention: None")
    print(f"   Innovation: Standard SSD approach")
    
    print(f"\nDifference:")
    print(f"   Additional parameters: {comparison['difference']['parameter_increase']:,}")
    print(f"   Percentage increase: {comparison['difference']['percentage_increase']:.2f}%")
    print("=" * 50)
    
    return comparison

# Example usage for thesis validation
if __name__ == "__main__":
    print("Testing Baseline SSD for thesis comparison...")
    
    # Test model creation
    baseline = create_baseline_ssd(
        num_classes=3,  # Example: background + 2 road hazard classes
        pretrained=True,
        device='cpu'  # Use CPU for testing
    )
    
    # Test forward pass
    test_input = torch.randn(2, 3, 320, 320)
    
    baseline.eval()
    with torch.no_grad():
        cls_logits, box_deltas, anchors = baseline(test_input)
        
        print(f"\n✅ Baseline SSD forward pass successful:")
        print(f"   Input: {test_input.shape}")
        print(f"   Classification: {cls_logits.shape}")
        print(f"   Box deltas: {box_deltas.shape}")
        print(f"   Anchors: {anchors.shape}")
    
    print("\n🎓 Baseline SSD ready for thesis comparison!")