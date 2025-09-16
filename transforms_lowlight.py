# transforms_lowlight.py
# FocusNet: SSD + MobileNetV3 + CBAM for Low-Light Road Hazard Detection
# Image preprocessing pipeline aligned with FocusNet architecture
# Input: Raw nighttime road images (0.5-10 lux) → MobileNetV3 backbone → CBAM → Multi-scale detection

import torch
import torchvision.transforms as T

class FocusNetTransforms:
    """
    Transform pipeline specifically designed for FocusNet architecture:
    Input Image → Preprocessing → MobileNetV3 Backbone → CBAM Attention → SSD Detection Head
    """
    
    @staticmethod
    def get_focusnet_input_size():
        """Standard input size for FocusNet architecture"""
        return 320
    
    @staticmethod
    def get_mobilenetv3_normalization():
        """ImageNet normalization parameters optimized for MobileNetV3 backbone"""
        return {
            'mean': [0.485, 0.456, 0.406],  # ImageNet RGB means
            'std': [0.229, 0.224, 0.225]   # ImageNet RGB stds
        }

def get_focusnet_train_transforms(img_size=320):
    """
    FocusNet training transforms - minimal preprocessing for clean architecture evaluation.
    
    Pipeline: Raw Image → Resize → Tensor → Normalize → MobileNetV3 → CBAM → SSD Head
    
    Args:
        img_size: Input size for FocusNet (default: 320x320)
    
    Returns:
        Transform pipeline for training FocusNet
    """
    norm_params = FocusNetTransforms.get_mobilenetv3_normalization()
    
    return T.Compose([
        T.Resize((img_size, img_size)),  # Resize for consistent input to MobileNetV3
        T.ToTensor(),                    # Convert PIL Image to tensor [0,1] range
        T.Normalize(                     # ImageNet normalization for MobileNetV3
            mean=norm_params['mean'], 
            std=norm_params['std']
        ),
    ])

def get_focusnet_eval_transforms(img_size=320):
    """
    FocusNet evaluation transforms for testing on RAW nighttime images.
    
    Purpose: Test model's inherent low-light detection capability through:
    - MobileNetV3 backbone feature extraction
    - CBAM attention mechanism focus
    - Multi-scale SSD detection
    
    Used for thesis validation:
    - Problem 1: Baseline SSD performance on raw images
    - Problem 2: FocusNet performance on raw images  
    - Problem 3: Comparative analysis (Wilcoxon test)
    
    Args:
        img_size: Input size for FocusNet (default: 320x320)
        
    Returns:
        Transform pipeline for evaluating FocusNet on raw low-light images
    """
    norm_params = FocusNetTransforms.get_mobilenetv3_normalization()
    
    return T.Compose([
        T.Resize((img_size, img_size)),  # Standard input for architecture comparison
        T.ToTensor(),                    # Preserve raw image characteristics
        T.Normalize(                     # MobileNetV3 pretrained normalization
            mean=norm_params['mean'], 
            std=norm_params['std']
        ),
    ])

def get_train_transforms(img_size=320):
    """Alias for FocusNet training transforms"""
    return get_focusnet_train_transforms(img_size)

def get_val_transforms(img_size=320):
    """Alias for FocusNet validation transforms"""
    return get_focusnet_eval_transforms(img_size)

def get_eval_transforms_raw(img_size=320):
    """Alias for raw image evaluation"""
    return get_focusnet_eval_transforms(img_size)

def get_thesis_test_transforms(img_size=320):
    """
    Thesis-specific transforms for FocusNet comparative analysis.
    
    Architecture Flow:
    Raw Night Image (0.5-10 lux) → Preprocessing → MobileNetV3 Backbone → 
    CBAM Attention → Multi-scale Feature Maps → SSD Detection Head → 
    Bounding Boxes + Classifications
    
    Validates: CBAM + MobileNetV3 architectural improvements over baseline SSD
    """
    return get_focusnet_eval_transforms(img_size)

# Architecture-specific utility functions
def get_focusnet_input_specs():
    """
    FocusNet architecture input specifications
    
    Returns:
        dict: Input specifications for FocusNet model
    """
    return {
        'input_size': (320, 320),
        'channels': 3,
        'normalization': FocusNetTransforms.get_mobilenetv3_normalization(),
        'backbone': 'MobileNetV3',
        'attention': 'CBAM',
        'detection_head': 'SSD',
        'target_conditions': 'Low-light (0.5-10 lux)',
        'augmentation_source': 'Roboflow (dataset-level)'
    }
