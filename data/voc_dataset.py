import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.voc_parser import parse_voc_xml

class VOCDataset(Dataset):
    def __init__(self, root, split_file, class_map, transforms=None):
        self.root = root
        self.transforms = transforms
        self.class_map = class_map

        with open(split_file) as f:
            self.image_ids = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_path = os.path.join(self.root, "images", f"{img_id}.jpg")
        ann_path = os.path.join(self.root, "annotations", f"{img_id}.xml")

        image = Image.open(img_path).convert("RGB")
        boxes, labels = parse_voc_xml(ann_path, self.class_map)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
