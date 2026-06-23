import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataset import CamVidDataset
from model import UNet
from evaluate import IoUMetric

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR   = "../data/CamVid"
EPOCHS     = 40
LR         = 1e-3
BATCH_SIZE = 8

def run_epoch(loader, model, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    metric = IoUMetric(num_classes=11, ignore_index=11)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            metric.update(torch.argmax(logits, dim=1), masks)

    return total_loss / len(loader), metric.compute()[0]


if __name__ == "__main__":
    train_loader = DataLoader(CamVidDataset(ROOT_DIR, "train"), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(CamVidDataset(ROOT_DIR, "val"),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = UNet(in_channels=3, num_classes=12).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=11)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = run_epoch(train_loader, model, criterion, optimizer)
        val_loss,   val_iou   = run_epoch(val_loader,   model, criterion)
        scheduler.step(val_loss)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_camvid.pth")
            saved = "  <- best saved"

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Train IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}  Val IoU: {val_iou:.4f}{saved}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
