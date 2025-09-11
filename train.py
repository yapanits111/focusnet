# train.py
import torch
from torch.utils.data import DataLoader
from detector import SSD_CBAM_MNV3
from ssd_head import SSDLoss
from transforms_lowlight import get_train_transforms, get_val_transforms

# You must implement your Dataset returning:
# image: PIL.Image
# target: dict(boxes=[G,4] in xyxy normalized 0..1, labels=[G] int (1..C-1))
class YourHazardDataset:
    def __init__(self, root, ann, transforms=None):
        self.transforms = transforms
        # load your image paths and annotations here
        ...
    def __len__(self): ...
    def __getitem__(self, idx):
        # return img, {'boxes': tensor([G,4]), 'labels': tensor([G])}
        ...

def collate(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)

def train_one_epoch(model, loss_fn, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = images.to(device)
        batch_targets = []
        for t in targets:
            bt = {'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)}
            batch_targets.append(bt)
        cls_logits, box_deltas, anchors = model(images)
        loss = loss_fn(cls_logits, box_deltas, anchors, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loss_fn, loader, device):
    model.eval()
    total = 0.0
    for images, targets in loader:
        images = images.to(device)
        batch_targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]
        cls_logits, box_deltas, anchors = model(images)
        loss = loss_fn(cls_logits, box_deltas, anchors, batch_targets)
        total += loss.item()
    return total / len(loader)

def main():
    num_classes = 1 + 4  # background + 4 hazard types (edit as needed)
    img_size = 320
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SSD_CBAM_MNV3(num_classes=num_classes, img_size=img_size).to(device)
    loss_fn = SSDLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    train_ds = YourHazardDataset(root='data/train', ann='data/train.json', transforms=get_train_transforms(img_size))
    val_ds   = YourHazardDataset(root='data/val', ann='data/val.json', transforms=get_val_transforms(img_size))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

    best_val = float('inf')
    for epoch in range(50):
        tr = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        vl = evaluate(model, loss_fn, val_loader, device)
        print(f'Epoch {epoch+1}: train {tr:.4f}  val {vl:.4f}')
        if vl < best_val:
            best_val = vl
            torch.save({'model': model.state_dict()}, 'ssd_cbam_mnv3_lowlight.pt')

if __name__ == '__main__':
    main()
