from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
