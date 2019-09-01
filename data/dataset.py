from torchvision import transforms


def get_train_transforms():
    return transforms.Compose([
        transforms.RandomPerspective(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

