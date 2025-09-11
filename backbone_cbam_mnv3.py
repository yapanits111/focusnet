# backbone_cbam_mnv3.py
import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from cbam import CBAM

class MNV3BackboneWithCBAM(nn.Module):
    """
    Extracts multi-scale features for SSD from MobileNetV3-Large.
    Outputs feature maps at strides ~16 and ~32 plus extra conv features.
    """
    def __init__(self, pretrained=True, cbam_positions=(6, 12, 15), freeze_bn=False):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.stem = nn.Sequential(base.features[0])  # stride 2

        # Split stages to tap features
        self.stage1 = nn.Sequential(*base.features[1:6])   # ends around stride 4/8
        self.stage2 = nn.Sequential(*base.features[6:12])  # mid-level
        self.stage3 = nn.Sequential(*base.features[12:])   # high-level (ends stride 16)

        # Inject CBAM at chosen depths (simple, robust placement)
        self.cbam1 = CBAM(self._out_channels(self.stage1))
        self.cbam2 = CBAM(self._out_channels(self.stage2))
        self.cbam3 = CBAM(self._out_channels(self.stage3))

        # Extra SSD layers to get more scales (strides ~32, ~64)
        self.extra1 = nn.Sequential(
            nn.Conv2d(self._out_channels(self.stage3), 256, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # CBAM in extra layers
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(256)

        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    @staticmethod
    def _out_channels(seq):
        # infer output channels by scanning last conv in a sequential block
        ch = None
        for m in seq.modules():
            if isinstance(m, nn.Conv2d):
                ch = m.out_channels
        return ch

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x); x1 = self.cbam1(x1)  # low/mid
        x2 = self.stage2(x1); x2 = self.cbam2(x2) # mid
        x3 = self.stage3(x2); x3 = self.cbam3(x3) # high (stride ~16)

        x4 = self.extra1(x3); x4 = self.cbam4(x4) # stride ~32
        x5 = self.extra2(x4); x5 = self.cbam5(x5) # stride ~64

        # choose a set of 4 feature maps for SSD head
        return [x2, x3, x4, x5]

    def out_channels_list(self):
        dummy = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            feats = self.forward(dummy)
        return [f.shape[1] for f in feats]
