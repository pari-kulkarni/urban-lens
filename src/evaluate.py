import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CamVidDataset
from model import UNet

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "../data/CamVid"

CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"
]

class IoUMetric:
    def __init__(self, num_classes=11, ignore_index=11):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes)
        self.union        = torch.zeros(num_classes)

    def reset(self):
        self.intersection.zero_()
        self.union.zero_()

    def update(self, preds, targets):
        preds, targets = preds.cpu(), targets.cpu()
        valid = targets != self.ignore_index
        for c in range(self.num_classes):
            pred_c   = (preds == c) & valid
            target_c = (targets == c) & valid
            self.intersection[c] += (pred_c & target_c).sum().item()
            self.union[c]        += (pred_c | target_c).sum().item()

    def compute(self):
        iou_per_class = self.intersection / (self.union + 1e-10)
        present       = self.union > 0
        mean_iou      = iou_per_class[present].mean().item() if present.any() else 0.0
        return mean_iou, iou_per_class


if __name__ == "__main__":
    test_loader = DataLoader(CamVidDataset(ROOT_DIR, "test"), batch_size=8, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, num_classes=12).to(DEVICE)
    model.load_state_dict(torch.load("best_unet_camvid.pth", map_location=DEVICE))
    model.eval()

    criterion  = nn.CrossEntropyLoss(ignore_index=11)
    metric     = IoUMetric(num_classes=11, ignore_index=11)
    total_loss = 0.0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits      = model(imgs)
            total_loss += criterion(logits, masks).item()
            metric.update(torch.argmax(logits, dim=1), masks)

    mean_iou, per_class = metric.compute()
    print(f"Test Loss : {total_loss / len(test_loader):.4f}")
    print(f"Mean IoU  : {mean_iou:.4f}\n")
    print(f"{'Class':<14} {'IoU':>6}")
    print("-" * 22)
    for name, iou in zip(CLASS_NAMES, per_class):
        print(f"{name:<14} {iou:.4f}  {'█' * int(iou * 20)}")
