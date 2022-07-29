import torch
import scipy.ndimage

import numpy as np
import pandas as pd

from pathlib import Path

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader




class DeepLenstronomyDataset(Dataset):
    """ The DeepLenstronomyDataset class.

    Args:
        Dataset: torch.utils.data.Dataset class
    """
    def __init__(self, target_keys_weights, root_dir, use_train=True, transform=None, target_transform=None):
        """ Initialize the class.

        Args:
            target_keys (list): list of targets (Y)
            root_dir (pathlib.Path object): dir of dataset
            use_train (bool, optional): True: training set. False: valiation set. Defaults to True.
            transform (torchvision.transforms, optional): transforms for images (X). Defaults to None.
            target_transform (torchvision.transforms, optional): transforms for targets (Y). Defaults to None.
        """
        self.target_keys_weights = target_keys_weights
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_train = use_train  # training set or test set
        if self.use_train:
            self.df = pd.read_csv(Path(f"{self.root_dir}/metadata_train_scaled.csv"))
        else:
            self.df = pd.read_csv(Path(f"{self.root_dir}/metadata_test_scaled.csv"))

    def __getitem__(self, index):
        if "img_name" in self.df.keys():
            img_name = self.df['img_name'].values[index]
        else:
            img_name = self.df['img_path'].values[index][-13:]
            print("img_name does not exist in meta csv so hard code by img_path instead.")
        
        img_path = Path(f"{self.root_dir}/{img_name}")
        img = np.load(img_path)

        ori_img_pixel = img.shape[0]
        image_channel = 3
        image_pixel = 224

        img = scipy.ndimage.zoom(img, image_pixel / ori_img_pixel, order=1)
        image = np.zeros((image_channel, image_pixel, image_pixel))

        for i in range(image_channel):
            image[i, :, :] += img

        target_dict = {key: self.df[key].iloc[[index]].values for key in self.target_keys_weights}

        return image, target_dict
        
    def __len__(self):
        return self.df.shape[0]




def get_train_test_datasets(CONFIG):
    """ Create DeepLenstronomyDataset objects.

    Args:
        CONFIG (dict): CONFIG

    Returns:
        train_dataset (DeepLenstronomyDataset): training dataset
        test_dataset (DeepLenstronomyDataset): testing dataset
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(), # scale to [0,1] and convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = torch.Tensor

    train_dataset = DeepLenstronomyDataset(
        CONFIG['target_keys_weights'],
        CONFIG['dataset_folder'], 
        use_train=True, 
        transform=data_transform, 
        target_transform=target_transform,
    )
    test_dataset = DeepLenstronomyDataset(
        CONFIG['target_keys_weights'],
        CONFIG['dataset_folder'], 
        use_train=False, 
        transform=data_transform, 
        target_transform=target_transform,
    )
    print("Number of train samples =", train_dataset.__len__())
    print("Number of test samples =", test_dataset.__len__())
    print(" ")
    return train_dataset, test_dataset


def get_train_test_dataloaders(batch_size, train_dataset, test_dataset):
    """ Convert Datasets into DataLoaders.

    Args:
        batch_size (int): batch size
        train_dataset (Dataset object): DeepLenstronomyDataset
        test_dataset (Dataset object): DeepLenstronomyDataset

    Returns:
        train_loader (DataLoader object): training DataLoader
        test_loader (DataLoader object): testing DataLoader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=True,
    )
    return train_loader, test_loader


# Below are for test set. Originally we only split into 2 sets: train and valid. 
# Now we have an extra set as the test set. I wrote the code below quickly so there 
# might have some code duplication or bad logic. Maybe can break down sets granularly.


class TestsetDeepLenstronomyDataset(DeepLenstronomyDataset):
    """ The TestsetDeepLenstronomyDataset class inherated from DeepLenstronomyDataset.

    Args:
        DeepLenstronomyDataset: DeepLenstronomyDataset class
    """
    def __init__(self, target_keys_weights, root_dir, transform=None, target_transform=None):
        """ Initialize the class.

        Args:
            target_keys (list): list of targets (Y)
            root_dir (pathlib.Path object): dir of dataset
            use_train (bool, optional): True: training set. False: testing set. Defaults to True.
            transform (torchvision.transforms, optional): transforms for images (X). Defaults to None.
            target_transform (torchvision.transforms, optional): transforms for targets (Y). Defaults to None.
        """
        self.target_keys_weights = target_keys_weights
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv(Path(f"{self.root_dir}/metadata_scaled.csv"))




def get_test_dataset(CONFIG):
    """ Create DeepLenstronomyDataset objects.

    Args:
        CONFIG (dict): CONFIG

    Returns:
        train_dataset (DeepLenstronomyDataset): training dataset
        test_dataset (DeepLenstronomyDataset): testing dataset
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(), # scale to [0,1] and convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = torch.Tensor

    test_dataset = TestsetDeepLenstronomyDataset(
        CONFIG['target_keys_weights'],
        CONFIG['dataset_folder'], 
        transform=data_transform, 
        target_transform=target_transform,
    )
    print("Number of test samples =", test_dataset.__len__())
    print(" ")
    return test_dataset


def get_test_dataloader(batch_size, test_dataset):
    """ Convert Datasets into DataLoaders.

    Args:
        batch_size (int): batch size
        train_dataset (Dataset object): DeepLenstronomyDataset
        test_dataset (Dataset object): DeepLenstronomyDataset

    Returns:
        train_loader (DataLoader object): training DataLoader
        test_loader (DataLoader object): testing DataLoader
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False,
    )
    return test_loader
