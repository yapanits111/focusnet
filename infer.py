# infer.py
import torch
import torchvision.ops as ops
from detector import SSD_CBAM_MNV3
from ssd_head import cxcywh_to_xyxy
from transforms_lowlight import get_val_transforms

@torch.no_grad()
def predict(model, image_pil, score_thresh=0.4, nms_thresh=0.45, topk=200, class_names=None):
    tr = get_val_transforms(model.img_size)
    img_t = tr(image_pil).unsqueeze(0).to(next(model.parameters()).device)

    cls_logits, box_deltas, anchors = model(img_t)
    probs = cls_logits.softmax(dim=-1)[0]  # [A,C]
    scores, labels = probs.max(dim=-1)     # [A]
    # decode boxes
    from ssd_head import SSDLoss
    decoder = SSDLoss()
    boxes_cxcywh = decoder.decode(anchors, box_deltas[0])
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh).clamp(0, 1)

    keep = scores > score_thresh
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    labels = labels[keep]

    # remove background
    fg = labels > 0
    boxes_xyxy = boxes_xyxy[fg]
    scores = scores[fg]
    labels = labels[fg]

    if boxes_xyxy.numel() == 0:
        return []

    # convert to pixel coords if needed; here keep as relative [0,1]
    keep_idx = ops.nms(boxes_xyxy, scores, nms_thresh)
    keep_idx = keep_idx[:topk]

    results = []
    for b, s, l in zip(boxes_xyxy[keep_idx], scores[keep_idx], labels[keep_idx]):
        cls_name = class_names[l.item()] if class_names else str(l.item())
        results.append({
            'bbox_xyxy_rel': b.tolist(),
            'score': float(s.item()),
            'label': int(l.item()),
            'name': cls_name
        })
    return results
