import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super(RPN, self).__init__()

        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.cls_logits = nn.Conv2d(256, num_anchors * 2, 1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, x):
        t = F.relu(self.conv(x))
        logits = self.cls_logits(t)
        bbox_regs = self.bbox_pred(t)
        return logits, bbox_regs
