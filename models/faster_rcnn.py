import torch
import torch.nn as nn
from models.backbone import SimpleBackbone
from models.rpn import RPN

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()

        self.backbone = SimpleBackbone()
        self.rpn = RPN(in_channels=128)

        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.cls_head = nn.Linear(1024, num_classes)
        self.box_head = nn.Linear(1024, num_classes * 4)

    def forward(self, images):
        features = self.backbone(images)
        rpn_logits, rpn_bbox = self.rpn(features)

        # NOTE: Simplified ROI pooling placeholder
        pooled = nn.functional.adaptive_max_pool2d(features, (7, 7))
        pooled = pooled.view(pooled.size(0), -1)

        x = nn.functional.relu(self.fc1(pooled))
        class_logits = self.cls_head(x)
        box_preds = self.box_head(x)

        return class_logits, box_preds, rpn_logits, rpn_bbox
