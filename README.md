# Road Scene Semantic Segmentation with U-Net

Pixel-wise semantic segmentation of urban road scenes trained on the CamVid dataset.
Built from scratch in PyTorch as a end-to-end computer vision project.

## Results

| Metric | Score |
|--------|-------|
| Mean IoU (test set) | **47.8%** |
| Test loss | 0.4254 |
| Best val IoU | 62.8% |

### Per-class IoU

| Class | IoU | Notes |
|-------|-----|-------|
| Sky | 91.4% | Large, uniform region — easy |
| Road | 90.7% | Large, uniform region — easy |
| Building | 74.9% | Strong |
| Pavement | 71.0% | Strong |
| Tree | 66.6% | Strong |
| Car | 63.3% | Moderate |
| SignSymbol | 19.3% | Small, rare — class imbalance |
| Pedestrian | 17.2% | Small, rare — class imbalance |
| Pole | 12.8% | Thin structure — hard geometry |
| Fence | 9.9% | Rare, thin — class imbalance |
| Bicyclist | 8.7% | Rarest class — class imbalance |

## Architecture

U-Net with 4 encoder/decoder levels and a bottleneck.

- **Encoder**: 4x DoubleConv blocks (3x3 conv → BN → ReLU × 2), followed by MaxPool
- **Bottleneck**: DoubleConv at 1024 channels
- **Decoder**: ConvTranspose2d upsampling + skip connections + DoubleConv
- **Output**: 1×1 conv → 12-channel logits (11 classes + 1 ignore)
- **Parameters**: ~31M

## Dataset

[CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) — Cambridge-driving Labeled Video Database.

- 367 training images, 101 validation, 233 test
- 11 semantic classes: Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist
- Images resized to 256×256, normalised with ImageNet mean/std

## Training

- **Loss**: CrossEntropyLoss (ignore_index=11 for unlabelled pixels)
- **Optimiser**: Adam, lr=1e-3
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Epochs**: 40
- **Batch size**: 8
- **Hardware**: Google Colab (T4 GPU)

## Key findings

The model demonstrates a clear class imbalance effect: classes that occupy many pixels
per image (Sky, Road) score 90%+ IoU, while rare or thin classes (Bicyclist, Fence, Pole)
score under 15%. Addressing this — via class-weighted loss, oversampling, or focal loss —
is a natural next step.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/road-segmentation
cd road-segmentation
pip install torch torchvision matplotlib pillow numpy
```

Download CamVid from the [SegNet tutorial repo](https://github.com/alexgkendall/SegNet-Tutorial).

## References

- Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (2015)
- Brostow et al., Semantic Object Classes in Video: A High-Definition Ground Truth Database (2009)
