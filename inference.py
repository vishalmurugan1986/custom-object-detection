import os
import torch
import cv2
import numpy as np
from PIL import Image

from models.faster_rcnn import FasterRCNN

# ----------------------------
# Configuration
# ----------------------------
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "background",
    1: "person",
    2: "car",
    3: "dog",
    4: "bicycle"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Load Model
# ----------------------------
model = FasterRCNN(num_classes=NUM_CLASSES)
model.load_state_dict(
    torch.load("model.pth", map_location=device)
)
model.to(device)
model.eval()

print("Model loaded successfully.")

# ----------------------------
# Load Image (PIL → OpenCV SAFE)
# ----------------------------
image_path = os.path.join("data", "images", "000001.jpg")

if not os.path.exists(image_path):
    raise RuntimeError(f"Image path does not exist: {image_path}")

# Robust image loading
pil_image = Image.open(image_path).convert("RGB")
image = np.array(pil_image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

orig_h, orig_w = image.shape[:2]

# ----------------------------
# Preprocess
# ----------------------------
image_resized = cv2.resize(image, (512, 512))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
image_tensor = image_tensor.float() / 255.0
image_tensor = image_tensor.unsqueeze(0).to(device)

# ----------------------------
# Inference
# ----------------------------
with torch.no_grad():
    class_logits, box_preds, _, _ = model(image_tensor)

pred_label = torch.argmax(class_logits, dim=1).item()

# Handle background prediction
if pred_label == 0:
    print("Model predicted background. No object detected.")
else:
    box_preds = box_preds.view(1, NUM_CLASSES, 4)
    pred_box = box_preds[0, pred_label].cpu().numpy()

    # ----------------------------
    # Scale boxes back to original image
    # ----------------------------
    x_scale = orig_w / 512
    y_scale = orig_h / 512

    xmin = int(pred_box[0] * x_scale)
    ymin = int(pred_box[1] * y_scale)
    xmax = int(pred_box[2] * x_scale)
    ymax = int(pred_box[3] * y_scale)

    # Clamp
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(orig_w - 1, xmax)
    ymax = min(orig_h - 1, ymax)

    label_name = CLASS_NAMES[pred_label]

    # ----------------------------
    # Draw Prediction
    # ----------------------------
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(
        image,
        label_name,
        (xmin, max(ymin - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# ----------------------------
# Save Output
# ----------------------------
output_path = "output_demo.jpg"
cv2.imwrite(output_path, image)

print("Inference completed successfully.")
print(f"Output saved at: {output_path}")

# ----------------------------
# Optional: Show window (use for screen recording)
# ----------------------------
cv2.imshow("Detection Output", image)
cv2.waitKey(3000)
cv2.destroyAllWindows()

