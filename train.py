import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.faster_rcnn import FasterRCNN
from data.voc_dataset import VOCDataset
from data.transforms import get_train_transforms

# ----------------------------
# Configuration
# ----------------------------
NUM_CLASSES = 5
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-3

CLASS_MAP = {
    "background": 0,
    "person": 1,
    "car": 2,
    "dog": 3,
    "bicycle": 4
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Dataset & DataLoader
# ----------------------------
dataset = VOCDataset(
    root="data",
    split_file="data/train.txt",
    class_map=CLASS_MAP,
    transforms=get_train_transforms()
)

def collate_fn(batch):
    return tuple(zip(*batch))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# ----------------------------
# Model
# ----------------------------
model = FasterRCNN(num_classes=NUM_CLASSES)
model.to(device)

# ----------------------------
# Losses & Optimizer
# ----------------------------
cls_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = torch.stack(images).to(device)

        # Take first object per image (simplification)
        gt_labels = []
        gt_boxes = []

        for t in targets:
            gt_labels.append(t["labels"][0])
            gt_boxes.append(t["boxes"][0])

        gt_labels = torch.stack(gt_labels).to(device)
        gt_boxes = torch.stack(gt_boxes).to(device)

        # Forward pass
        class_logits, box_preds, _, _ = model(images)

        # Classification loss
        cls_loss = cls_criterion(class_logits, gt_labels)

        # Bounding box regression loss
        box_preds = box_preds.view(box_preds.size(0), NUM_CLASSES, 4)
        pred_boxes = box_preds[torch.arange(len(gt_labels)), gt_labels]
        bbox_loss = bbox_criterion(pred_boxes, gt_boxes)

        loss = cls_loss + bbox_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# ----------------------------
# Save Model
# ----------------------------
save_path = os.path.join(os.getcwd(), "model.pth")
torch.save(model.state_dict(), save_path)

print(f"Training completed successfully.")
print(f"Model saved at: {save_path}")


