import glob
import os

import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import random


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


def generate_categories(path):
    """

    :param path:
    :return:
    """
    category_list = []
    categories = {}

    for directory, subdirectories, files in os.walk(path):

        for subdirectory in subdirectories:
            category_list.append(subdirectory)
            categories[subdirectory] = []

            for file in glob.glob(os.path.join(path, subdirectory, '*')):
                categories[subdirectory].append(file)

        return category_list, categories


class TripletDataSet(data.Dataset):

    def __init__(self, dataset_path, length, crop_size):
        self.length = length
        self.category_list, self.category_files = generate_categories(dataset_path)
        self.category_count = len(self.category_list)
        self.flip_transform = get_flip_transforms(crop_size)
        self.input_transform = get_train_transforms(crop_size)
        self.target_transform = get_target_transforms(crop_size)

    def _random_category(self):
        """

        :return:
        """
        return random.choice(self.category_list)

    def _random_file(self, category_name):
        """

        :param category_name:
        :return:
        """
        return random.choice(self.category_files[category_name])

    def _random_files(self):
        train = negative = [None, None]

        # We don't accept images from the same category, so loop until we are sure they wont match.
        while train == negative:
            train = self._random_category()
            negative = self._random_category()

        train_file = self._random_file(train)
        negative_file = self._random_file(negative)

        # Sample a random file from both train and target categories.
        return train_file, negative_file

    def __getitem__(self, index):
        train_file, negative_file = self._random_files()
        # Load the images required for the data.
        train = Image.open(train_file).convert('RGB')
        target = Image.open(negative_file).convert('RGB')
        flipped = train.copy()

        # Do the required transformations
        train = self.input_transform(train)
        flipped = self.flip_transform(flipped)
        target = self.target_transform(target)

        return torch.stack([train, flipped, target])

    def __len__(self):
        return self.length


def get_flip_transforms(crop_size):
    """

    :param crop_size:
    :return:
    """
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.RandomHorizontalFlip(1.0),
        transforms.ToTensor()
    ])


def get_train_transforms(crop_size):
    """

    :param crop_size:
    :return:
    """
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])


def get_target_transforms(crop_size):
    """

    :param crop_size:
    :return:
    """
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])
