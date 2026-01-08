import torch
import time
from torch.utils.data import DataLoader

from models.faster_rcnn import FasterRCNN
from data.voc_dataset import VOCDataset
from data.transforms import get_val_transforms
from eval_utils import compute_iou

CLASS_MAP = {
    "background": 0,
    "person": 1,
    "car": 2,
    "dog": 3,
    "bicycle": 4
}

NUM_CLASSES = 5
IOU_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = VOCDataset(
    root="data",
    split_file="data/val.txt",
    class_map=CLASS_MAP,
    transforms=get_val_transforms()
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Model
model = FasterRCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

true_positives = 0
false_positives = 0
false_negatives = 0

start_time = time.time()

with torch.no_grad():
    for image, target in loader:
        image = image.to(device)

        gt_box = target["boxes"][0]
        gt_label = target["labels"][0].item()

        class_logits, box_preds, _, _ = model(image)

        pred_label = torch.argmax(class_logits, dim=1).item()

        box_preds = box_preds.view(1, NUM_CLASSES, 4)
        pred_box = box_preds[0, pred_label].cpu()

        iou = compute_iou(pred_box, gt_box)

        if pred_label == gt_label and iou >= IOU_THRESHOLD:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1

end_time = time.time()

# Precision / Recall
precision = true_positives / max(true_positives + false_positives, 1)
recall = true_positives / max(true_positives + false_negatives, 1)

mAP_50 = precision  # simplified single-threshold AP

fps = len(dataset) / (end_time - start_time)

print(f"mAP@0.5: {mAP_50:.4f}")
print(f"FPS: {fps:.2f}")
