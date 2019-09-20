import os
from torch.utils import data
from torchvision import transforms
from PIL import Image


def get_images(root_dir):
    """
    Gets all the images from a directory recursively.
    :param root_dir: The starting directory to search for images.
    :return: A list of image files.
    """
    images = []
    for dir, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(dir, file))

    return images


class FolderDataSet(data.Dataset):

    def __init__(self, root_dir, crop_size):
        """
        Creates a new dataset of images from the provided root directory.
        :param root_dir: The folder to search for images.
        :param crop_size: The size to crop the images to.
        """
        super(FolderDataSet, self).__init__()
        self.files = get_images(root_dir)
        self.input_transform = get_train_transforms(crop_size)
        self.target_transform = get_train_transforms(crop_size)
        self.flip_transform = get_flip_transforms(crop_size)

    def __getitem__(self, index):
        train = Image.open(self.files[index]).convert('RGB')
        flipped = train.copy()

        self.input_transform(train)
        self.flip_transform(flipped)

        return train, flipped

    def __len__(self):
        return len(self.files)


def get_flip_transforms(crop_size):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor()
    ])


def get_train_transforms(crop_size):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])


def get_target_transforms(crop_size):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])


def get_train_dataset(root, crop_size):
    """
    Get the training dataset.
    :param root: The folder to search for images in.
    :param crop_size
    :return:
    """
    return FolderDataSet(
        root_dir=root,
        crop_size=crop_size
    )
