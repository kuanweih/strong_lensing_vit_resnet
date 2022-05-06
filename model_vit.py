""" Code taken from https://github.com/joshualin24/vit_strong_lensing/blob/master/vit.py 
    Originally created by 2022-4-4 neural modelworks by Joshua Yao-Yu Lin
    Modified by Kuan-Wei Huang
"""


import os
from matplotlib.pyplot import axis
import scipy.ndimage

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from transformers import ViTForImageClassification




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
            use_train (bool, optional): True: training set. False: testing set. Defaults to True.
            transform (torchvision.transforms, optional): transforms for images (X). Defaults to None.
            target_transform (torchvision.transforms, optional): transforms for targets (Y). Defaults to None.
        """
        self.target_keys_weights = target_keys_weights
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_train = use_train  # training set or test set
        if self.use_train:
            self.df = pd.read_csv(Path(f"{self.root_dir}/metadata_train.csv"))
        else:
            self.df = pd.read_csv(Path(f"{self.root_dir}/metadata_test.csv"))

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


def print_n_train_params(model):
    """ Print number of trainable parameters.

    Args:
        model (model object): presumably a ViT model
    """
    n = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Number of trainable parameters: {n}")


def prepare_data_and_target(data, target_dict, device):
    """ Prepare data (X) and target (Y) for a given batch.

    Args:
        data (np.array): image (X)
        target_dict (dict): targets (Y)
        device (torch.device): cpu or gpu

    Returns:
        data (torch.Tensor): image (X)
        target (torch.Tensor): targets(Y)
    """
    data = Variable(data.float()).to(device)
    for key, val in target_dict.items():
        target_dict[key] = Variable(val.float()).to(device)
    target = torch.cat([val for _, val in target_dict.items()], dim=1)
    return data, target




class CacheHistory:

    def __init__(self, cache_epoch):
        self.cache_epoch = cache_epoch




class CacheEpoch:

    def __init__(self):
        """ Initialize cache dict.

        Returns:
            cache (dict): initialized cache dict
        """
        self._cnt = 0
        self._cum_loss = 0.0
        self._cum_err_mean = 0
        self._cum_err_std = 0

        self.avg_loss = 0.0
        self.avg_err_mean = 0
        self.avg_err_std = 0

    def update_cache(self, pred, target, loss):
        """ Update cache dict.

        Args:
            pred ([type]): [description]
            target ([type]): [description]
            loss ([type]): [description]
        """
        err = (pred - target).cpu().detach().numpy()
        self._cum_err_mean += err.mean(axis=0)
        self._cum_err_std += err.std(axis=0)
        self._cum_loss += loss.item()
        self._cnt += 1

    def calc_avg_across_batches(self):
        self.avg_err_mean = self._cum_err_mean / self._cnt
        self.avg_err_std = self._cum_err_std / self._cnt
        self.avg_loss = self._cum_loss / self._cnt

    def print_loss_rms(self, epoch, train_test_str):
        """ Print loss and rms.

        Args:
            epoch ([type]): [description]
            train_test_str ([type]): [description]
            cache ([type]): [description]
        """
        avg_err_mean_print = np.array_str(self.avg_err_mean, precision=4)
        avg_err_std_print = np.array_str(self.avg_err_std, precision=4)
        print(f"epoch = {epoch}, {train_test_str}:")
        print(f"    loss (average over epoch) = {self.avg_loss:.4f}")
        print(f"    err_mean (average over epoch) = {avg_err_mean_print}")
        print(f"    err_std (average over epoch) = {avg_err_std_print}")


def calc_loss(pred, target, CONFIG, device):
    """ Average weighted mean squared loss per sample.

    Let 
        i = range(0, B): sample index in a batch with size B
        j = range(0, T): target index for T targets
        pred_(i, j): prediction for target j and sample i
        truth_(i, j): target for target j and sample i
        SE_(i, j): squared error for target j and sample i
        MSE_j: mean squared error for target j per sample
        w_j: weight for target j
        W = sum(w_j for j in range(T)): total weight

    SE_(i, j) = (pred_(i, j) - truth_(i, j))**2
    MSE_j = sum(SE_(i, j) for i in range(B)) / B
    Loss = sum(w_j * MSE_j for j in range(T)) / W
    
    Args:
        pred (torch.Tensor): prediction of a batch
        target (torch.Tensor): target of a batch
        CONFIG (dict): CONFIG
        device (torch.device): cpu or gpu

    Returns:
        [torch.Tensor]: Loss
    """
    weight = [w for _, w in CONFIG["target_keys_weights"].items()]
    weight = torch.tensor(weight, requires_grad=False).to(device)
    loss = torch.mean((pred - target)**2, axis=0)
    loss = torch.sum(weight * loss) / weight.sum()
    return loss


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


def prepare_vit_model(CONFIG):
    """ Prepare a fresh pretrained ViT model.

    Args:
        CONFIG (dict): CONFIG

    Returns:
        model (model object): a ViT model constructed based on CONFIG
    """
    out_features = len(CONFIG['target_keys_weights'])
    model = ViTForImageClassification.from_pretrained(CONFIG['pretrained_model_name'])
    print_n_train_params(model)
    model.classifier = nn.Linear(in_features=768, out_features=out_features, bias=True)
    print_n_train_params(model)
    print(" ")
    return model


def initialize_loss_history():
    """ Inintialize the loss history dict.

    Returns:
        (dict): empty loss history dict
    """
    return {
        'epoch': [],
        'batch_idx': [],
        'loss': [],
    }


def record_loss_history(history_dict, epoch, batch_idx, loss):
    """ Record loss to the history dict.

    Args:
        history_dict (dict): dict contains loss info
        epoch (int): current epoch
        batch_idx (int): current batch id 
        loss (): current loss

    Returns:
        history_dict (dict): the updated history dict
    """
    history_dict['epoch'].append(epoch)
    history_dict['batch_idx'].append(batch_idx)
    history_dict['loss'].append(loss.item())
    return history_dict


def save_loss_history(CONFIG, history_dict, which):
    """ Save loss history as npy.

    Args:
        CONFIG (dict): CONFIG
        history_dict (dict): dict contains loss info
        which (str): which loss history to be saved. 'train' or 'test'
    """
    fname = f"{CONFIG['dir_model_save']}/{CONFIG['model_file_name_prefix']}_{which}_loss_history.npy"
    np.save(fname, history_dict)

    # TODO: class
    # dd = np.load(fname, allow_pickle=True)
    # print(dd)


def save_config(CONFIG):
    fname = f"{CONFIG['dir_model_save']}/{CONFIG['model_file_name_prefix']}_CONFIG.npy"
    np.save(fname, CONFIG)


def train_model(CONFIG):
    """ Train a model based on parameters in CONFIG.

    Args:
        CONFIG (dict): model configuration dict
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Use device = {device}\n")

    if not os.path.exists(CONFIG['dir_model_save']):
        os.mkdir(CONFIG['dir_model_save'])

    save_config(CONFIG)

    # prepare data loaders
    train_dataset, test_dataset = get_train_test_datasets(CONFIG)
    train_loader, test_loader = get_train_test_dataloaders(CONFIG['batch_size'], train_dataset, test_dataset)

    # prepare model
    if CONFIG['new_vit_model']:
        model = prepare_vit_model(CONFIG)
        print(f"Use fresh pretrained model = {CONFIG['pretrained_model_name']}\n")
    else:
        model = torch.load(CONFIG['path_model_to_resume'])  
        print(f"Use our trained model = {CONFIG['path_model_to_resume']}\n")

    model.to(device) 
    print(f"Model cast to device = {model.device}\n")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['init_learning_rate'])

    best_test_accuracy = float("inf")  # to identify best model ever seen
    test_loss_history = initialize_loss_history()
    train_loss_history = initialize_loss_history()



    for epoch in range(CONFIG['epoch']):
        model.train()

        cache_train = CacheEpoch()

        for batch_idx, (data, target_dict) in enumerate(tqdm(train_loader, total=len(train_loader))):
            data, target = prepare_data_and_target(data, target_dict, device)
            optimizer.zero_grad()
            output = model(data)[0] 
            loss = calc_loss(output, target, CONFIG, device)

            cache_train.update_cache(output, target, loss)

            loss.backward()
            optimizer.step()

            if batch_idx % CONFIG['record_loss_every_num_batch'] == 0 and batch_idx != 0:
                train_loss_history = record_loss_history(train_loss_history, epoch, batch_idx, loss)


        cache_train.calc_avg_across_batches()
        cache_train.print_loss_rms(epoch, 'Train')


        with torch.no_grad():
            model.eval()
            cache_test = CacheEpoch()

            for batch_idx, (data, target_dict) in enumerate(test_loader):
                data, target = prepare_data_and_target(data, target_dict, device)
                pred = model(data)[0]
                loss = calc_loss(pred, target, CONFIG, device)

                cache_test.update_cache(pred, target, loss)

                if batch_idx % CONFIG['record_loss_every_num_batch'] == 0 and batch_idx != 0:
                    test_loss_history = record_loss_history(test_loss_history, epoch, batch_idx, loss)

            cache_test.calc_avg_across_batches()
            cache_test.print_loss_rms(epoch, 'Test')


            # TODO: get test from history instead of cache
            test_loss_per_batch = cache_test.avg_loss
            if test_loss_per_batch < best_test_accuracy:
                best_test_accuracy = test_loss_per_batch
                _dir = CONFIG['dir_model_save']
                _prefix = CONFIG['model_file_name_prefix']
                model_save_path = f"{_dir}/{_prefix}_epoch_{epoch}_testloss_{test_loss_per_batch:.6f}.mdl"
                torch.save(model, model_save_path)
                print(f"\nSave model to {model_save_path}\n")
    
        for which, history_dict in zip(["train", "test"], [train_loss_history, test_loss_history]):
            save_loss_history(CONFIG, history_dict, which)




if __name__ == '__main__':

    CONFIG = {
        'epoch': 2,
        'batch_size': 30,
        'new_vit_model': True,
        'pretrained_model_name': "google/vit-base-patch16-224", # for 'new_vit_model' = True
        'path_model_to_resume': Path(""), # for 'new_vit_model' = False
        'dataset_folder': Path("C:/Users/abcd2/Datasets/2022_icml_lens_sim/dev_256"),
        'dir_model_save': Path("C:/Users/abcd2/Downloads/tmp_dev_outputs"),
        'model_file_name_prefix': 'vit_dev',
        'init_learning_rate': 1e-4,
        'record_loss_every_num_batch': 2,
        'target_keys_weights': {
            "theta_E": 10, 
            "gamma": 1, 
            "center_x": 1, 
            "center_y": 1, 
            "e1": 1, 
            "e2": 1, 
            "gamma_ext": 1, 
            "psi_ext": 1, 
            "lens_light_R_sersic": 1, 
            "lens_light_n_sersic": 1,
        }
    }

    train_model(CONFIG)



