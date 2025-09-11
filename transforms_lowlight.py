# transforms_lowlight.py
import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class RandomGamma:
    def __init__(self, gamma_range=(0.5, 1.5)):
        self.gamma_range = gamma_range
    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        return TF.adjust_gamma(img, gamma)

class RandomCLAHE:
    # lightweight approximation using equalize; swap with OpenCV CLAHE for best results
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return TF.equalize(img)
        return img

def get_train_transforms(img_size=320):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        RandomGamma((0.4, 1.4)),
        RandomCLAHE(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def get_val_transforms(img_size=320):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
