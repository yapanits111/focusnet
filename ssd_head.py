# ssd_head.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_anchor_grid(img_size=320, fm_sizes=(40, 20, 10, 5), 
                     scales=(0.08, 0.16, 0.32, 0.48), 
                     aspect_ratios=((1.0, 2.0, 0.5), (1.0, 2.0, 0.5),
                                    (1.0, 2.0, 0.5), (1.0, 2.0, 0.5))):
    anchors = []
    for k, fm in enumerate(fm_sizes):
        s = scales[k]
        for i in range(fm):
            for j in range(fm):
                cx = (j + 0.5) / fm
                cy = (i + 0.5) / fm
                for ar in aspect_ratios[k]:
                    w = s * math.sqrt(ar)
                    h = s / math.sqrt(ar)
                    anchors.append([cx, cy, w, h])
                # extra scale between s and next s (SSD trick)
                s_next = scales[min(k + 1, len(scales) - 1)]
                s_prime = math.sqrt(s * s_next)
                anchors.append([cx, cy, s_prime, s_prime])
    return torch.tensor(anchors)  # [A,4] in cx,cy,w,h (relative)

class SSDHead(nn.Module):
    def __init__(self, in_channels_list, num_classes, num_anchors=(4,4,4,4)):
        super().__init__()
        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        self.num_classes = num_classes

        for ch, na in zip(in_channels_list, num_anchors):
            self.cls_heads.append(nn.Conv2d(ch, na * num_classes, 3, padding=1))
            self.box_heads.append(nn.Conv2d(ch, na * 4, 3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, feats):
        logits = []
        boxes = []
        for f, cls_head, box_head in zip(feats, self.cls_heads, self.box_heads):
            cls = cls_head(f).permute(0, 2, 3, 1).contiguous()
            box = box_head(f).permute(0, 2, 3, 1).contiguous()
            logits.append(cls.view(cls.size(0), -1, self.num_classes))
            boxes.append(box.view(box.size(0), -1, 4))
        return torch.cat(logits, dim=1), torch.cat(boxes, dim=1)

def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)

def box_iou(b1, b2):
    # b1: [N,4], b2:[M,4] in xyxy
    area1 = (b1[:,2]-b1[:,0]).clamp(0) * (b1[:,3]-b1[:,1]).clamp(0)
    area2 = (b2[:,2]-b2[:,0]).clamp(0) * (b2[:,3]-b2[:,1]).clamp(0)
    lt = torch.max(b1[:, None, :2], b2[:, :2])  # [N,M,2]
    rb = torch.min(b1[:, None, 2:], b2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[...,0]*wh[...,1]
    union = area1[:,None] + area2 - inter
    return inter / (union + 1e-6)

def match_anchors(anchors_xyxy, gt_boxes_xyxy, gt_labels, iou_thresh=0.5):
    A = anchors_xyxy.size(0)
    matched_labels = torch.zeros(A, dtype=torch.long)  # background=0
    matched_boxes = torch.zeros(A, 4)

    if gt_boxes_xyxy.numel() == 0:
        return matched_boxes, matched_labels

    ious = box_iou(anchors_xyxy, gt_boxes_xyxy)  # [A,G]
    best_gt_iou, best_gt_idx = ious.max(dim=1)

    # ensure each gt has at least one anchor
    best_anchor_iou, best_anchor_idx = ious.max(dim=0)
    matched_labels[best_anchor_idx] = gt_labels
    matched_boxes[best_anchor_idx] = gt_boxes_xyxy
    best_gt_iou[best_anchor_idx] = 1.0  # force positive

    pos_mask = best_gt_iou >= iou_thresh
    matched_labels[pos_mask] = gt_labels[best_gt_idx[pos_mask]]
    matched_boxes[pos_mask] = gt_boxes_xyxy[best_gt_idx[pos_mask]]
    return matched_boxes, matched_labels

class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance

    def encode(self, anchors_cxcywh, gt_cxcywh):
        g_cxcy = (gt_cxcywh[...,:2] - anchors_cxcywh[...,:2]) / (self.center_variance * anchors_cxcywh[...,:2])
        g_wh = torch.log(gt_cxcywh[...,2:] / anchors_cxcywh[...,2:]) / self.size_variance
        return torch.cat([g_cxcy, g_wh], dim=-1)

    def decode(self, anchors_cxcywh, deltas):
        cxcy = deltas[...,:2] * self.center_variance * anchors_cxcywh[...,:2] + anchors_cxcywh[...,:2]
        wh = torch.exp(deltas[...,2:] * self.size_variance) * anchors_cxcywh[...,2:]
        return torch.cat([cxcy, wh], dim=-1)

    def forward(self, cls_logits, box_reg, anchors_cxcywh, targets):
        """
        targets: list of dicts: {'boxes': [G,4] in xyxy (0-1), 'labels': [G] in 1..C-1}
        """
        B, A, C = cls_logits.size()
        anchors_xyxy = cxcywh_to_xyxy(anchors_cxcywh)

        cls_targets = []
        box_targets = []
        for b in range(B):
            gt = targets[b]
            mb, ml = match_anchors(anchors_xyxy, gt['boxes'], gt['labels'])
            box_targets.append(xyxy_to_cxcywh(mb))
            cls_targets.append(ml)
        box_targets = torch.stack(box_targets, dim=0)  # [B,A,4]
        cls_targets = torch.stack(cls_targets, dim=0)  # [B,A]

        # box regression loss (L1 on positives)
        pos_mask = cls_targets > 0
        box_pred_pos = box_reg[pos_mask]
        box_tgt_pos = self.encode(anchors_cxcywh[pos_mask], box_targets[pos_mask])
        loc_loss = F.smooth_l1_loss(box_pred_pos, box_tgt_pos, reduction='sum')

        # classification loss with hard negative mining
        cls_loss_all = F.cross_entropy(cls_logits.view(-1, C), cls_targets.view(-1), reduction='none').view(B, A)
        num_pos = pos_mask.sum(dim=1).clamp(min=1)
        num_neg = self.neg_pos_ratio * num_pos

        cls_loss = torch.zeros(1, device=cls_logits.device)
        for b in range(B):
            loss_b = cls_loss_all[b]
            pos_b = pos_mask[b]
            neg_b = (~pos_b)
            # top-k negatives
            num_neg_b = int(num_neg[b].item())
            neg_loss_b = loss_b[neg_b]
            if neg_loss_b.numel() > 0:
                topk_neg = torch.topk(neg_loss_b, k=min(num_neg_b, neg_loss_b.numel())).values
                cls_loss += (loss_b[pos_b].sum() + topk_neg.sum())
            else:
                cls_loss += loss_b[pos_b].sum()

        N = num_pos.sum().clamp(min=1).float()
        return (loc_loss + cls_loss) / N
