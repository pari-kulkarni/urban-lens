import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from dataset import CamVidDataset
from model import UNet

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "../data/CamVid"

CLASS_NAMES  = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"
]
CLASS_COLORS = np.array([
    [128,128,128],[128,0,0],[192,192,128],[128,64,128],[60,20,220],
    [128,128,0],[192,128,128],[64,64,128],[64,0,128],[64,64,0],[0,128,192]
], dtype=np.uint8)

def decode_mask(mask_tensor):
    mask = mask_tensor.cpu().numpy()
    rgb  = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb

def denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t.cpu() * std + mean).clamp(0,1).permute(1,2,0).numpy()


if __name__ == "__main__":
    test_loader = DataLoader(CamVidDataset(ROOT_DIR, "test"), batch_size=8, shuffle=False)

    model = UNet(in_channels=3, num_classes=12).to(DEVICE)
    model.load_state_dict(torch.load("best_unet_camvid.pth", map_location=DEVICE))
    model.eval()

    imgs, masks = next(iter(test_loader))
    with torch.no_grad():
        preds = torch.argmax(model(imgs.to(DEVICE)), dim=1)

    N = 4
    fig, axes = plt.subplots(N, 3, figsize=(13, N * 3.5))
    fig.suptitle("UrbanLens — predictions on CamVid test set", fontsize=14)

    for ax, col in zip(axes[0], ["Input image", "Ground truth", "Prediction"]):
        ax.set_title(col, fontsize=11)

    for i in range(N):
        axes[i][0].imshow(denorm(imgs[i]))
        axes[i][1].imshow(decode_mask(masks[i]))
        axes[i][2].imshow(decode_mask(preds[i]))
        for ax in axes[i]:
            ax.axis("off")

    patches = [mpatches.Patch(color=c/255, label=n)
               for n, c in zip(CLASS_NAMES, CLASS_COLORS)]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig("../outputs/predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to outputs/predictions.png")
