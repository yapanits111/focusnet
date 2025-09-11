# detector.py
import torch
import torch.nn as nn
from backbone_cbam_mnv3 import MNV3BackboneWithCBAM
from ssd_head import SSDHead, make_anchor_grid, cxcywh_to_xyxy

class SSD_CBAM_MNV3(nn.Module):
    def __init__(self, num_classes, img_size=320):
        super().__init__()
        self.backbone = MNV3BackboneWithCBAM(pretrained=True)
        chs = self.backbone.out_channels_list()
        self.head = SSDHead(chs, num_classes)
        # infer fm sizes from a dummy forward
        with torch.no_grad():
            x = torch.zeros(1,3,img_size,img_size)
            feats = self.backbone(x)
            fm_sizes = [f.shape[-1] for f in feats]
        # anchor counts per loc must match head num_anchors; here 4 per loc by design
        scales = (0.08, 0.16, 0.32, 0.48)
        self.register_buffer('anchors', make_anchor_grid(img_size, tuple(fm_sizes), scales))
        self.img_size = img_size
        self.num_classes = num_classes

    def forward(self, images, targets=None):
        feats = self.backbone(images)
        cls_logits, box_deltas = self.head(feats)
        return cls_logits, box_deltas, self.anchors
