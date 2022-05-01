""" Code taken from https://github.com/joshualin24/vit_strong_lensing/blob/master/vit.py 
    Originally created by 2022-4-4 neural modelworks by Joshua Yao-Yu Lin
    Modified by Kuan-Wei Huang
"""


import os
import datetime
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




class DeepLenstronomyDataset(Dataset):  # torch.utils.data.Dataset
    
    def __init__(self, target_keys, root_dir, use_train=True, transform=None, target_transform=None):
        self.target_keys = target_keys
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

        target_dict = {key: self.df[key].iloc[[index]].values for key in self.target_keys}

        return image, target_dict
        
    def __len__(self):
        return self.df.shape[0]


def print_n_train_params(model):
    n = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Number of trainable parameters: {n}")


def prepare_data_and_target(data, target_dict, device):
    data = Variable(data.float()).to(device)
    for key, val in target_dict.items():
        target_dict[key] = Variable(val.float()).to(device)
    target = torch.cat([val for _, val in target_dict.items()], dim=1)
    return data, target


def calc_avg_rms(cache):
    avg_rms = cache['total_rms'] / cache['total_counter']
    avg_rms = avg_rms.cpu()
    avg_rms = (avg_rms.data).numpy()
    cache['avg_rms'] = avg_rms
    #TODO
    # for i in range(len(avg_rms)):
    #     tb.add_scalar(f"rms {i + 1}", avg_rms[i])
    return cache
    

def print_loss_rms(epoch, train_test_str, cache):
    loss_per_batch = cache['total_loss'] / cache['total_counter']
    avg_rms_for_print = np.array_str(cache['avg_rms'], precision=4)
    print(f"epoch = {epoch}, {train_test_str}:")
    print(f"    loss (average per batch wise) = {loss_per_batch:.4e}")
    print(f"    RMS (average per batch wise) = {avg_rms_for_print}")


def update_cache(cache, pred, target, loss):
    square_diff = pred - target  
    cache['total_rms'] += square_diff.std(dim=0)
    cache['total_loss'] += loss.item()
    cache['total_counter'] += 1
    return cache


def initialize_cache():
    cache = {
        'total_loss': 0.0,
        'total_counter': 0,
        'total_rms': 0,
    }
    return cache


def calc_loss(loss_fn, pred, target):
    loss_theta_E = loss_fn(100*pred[0], 100*target[0])  #TODO: this is hard coded
    loss_others = loss_fn(pred, target)
    loss = loss_theta_E + loss_others
    return loss


def get_train_test_datasets(CONFIG):
    data_transform = transforms.Compose([
        transforms.ToTensor(), # scale to [0,1] and convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = torch.Tensor

    train_dataset = DeepLenstronomyDataset(
        CONFIG['target_keys'],
        CONFIG['dataset_folder'], 
        use_train=True, 
        transform=data_transform, 
        target_transform=target_transform,
    )
    test_dataset = DeepLenstronomyDataset(
        CONFIG['target_keys'],
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
    out_features = len(CONFIG['target_keys'])
    model = ViTForImageClassification.from_pretrained(CONFIG['pretrained_model_name'])
    print_n_train_params(model)
    model.classifier = nn.Linear(in_features=768, out_features=out_features, bias=True)
    print_n_train_params(model)
    print(" ")
    return model


def initialize_loss_history():
    return {
        'epoch': [],
        'batch_idx': [],
        'loss': [],
    }


def record_loss_history(history_dict, epoch, batch_idx, loss):
    history_dict['epoch'].append(epoch)
    history_dict['batch_idx'].append(batch_idx)
    history_dict['loss'].append(loss.item())
    return history_dict


def save_loss_history(CONFIG, history_dict, which):
    fname = f"{CONFIG['dir_model_save']}/{CONFIG['model_file_name_prefix']}_{which}_loss_history.npy"
    np.save(fname, history_dict)


def train_model(CONFIG):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Use device = {device}\n")

    if not os.path.exists(CONFIG['dir_model_save']):
        os.mkdir(CONFIG['dir_model_save'])

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

    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['init_learning_rate'])

    best_test_accuracy = float("inf")  # to identify best model ever seen
    test_loss_history = initialize_loss_history()
    train_loss_history = initialize_loss_history()

    for epoch in range(CONFIG['epoch']):
        model.train()
        cache_train = initialize_cache()

        for batch_idx, (data, target_dict) in enumerate(tqdm(train_loader, total=len(train_loader))):
            data, target = prepare_data_and_target(data, target_dict, device)
            optimizer.zero_grad()
            output = model(data)[0] 
            loss = calc_loss(loss_fn, output, target)
            cache_train = update_cache(cache_train, output, target, loss)
            loss.backward()
            optimizer.step()

            if batch_idx % CONFIG['record_loss_every_num_batch'] == 0 and batch_idx != 0:
                train_loss_history = record_loss_history(train_loss_history, epoch, batch_idx, loss)


        cache_train = calc_avg_rms(cache_train)
        print_loss_rms(epoch, 'Train', cache_train)


        with torch.no_grad():
            model.eval()
            cache_test = initialize_cache()

            for batch_idx, (data, target_dict) in enumerate(test_loader):
                data, target = prepare_data_and_target(data, target_dict, device)
                pred = model(data)[0]
                loss = calc_loss(loss_fn, pred, target)
                cache_test = update_cache(cache_test, pred, target, loss)

                if batch_idx % CONFIG['record_loss_every_num_batch'] == 0 and batch_idx != 0:
                    test_loss_history = record_loss_history(test_loss_history, epoch, batch_idx, loss)

            cache_test = calc_avg_rms(cache_test)
            print_loss_rms(epoch, 'Test', cache_test)

            test_loss_per_batch = cache_test['total_loss'] / cache_test['total_counter']
            if test_loss_per_batch < best_test_accuracy:
                best_test_accuracy = test_loss_per_batch
                time_stamp = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
                _dir = CONFIG['dir_model_save']
                _prefix = CONFIG['model_file_name_prefix']
                model_save_path = f"{_dir}/{_prefix}_{time_stamp}_testloss_{test_loss_per_batch:.4e}.mdl"
                torch.save(model, model_save_path)
                print(f"\nSave model to {model_save_path}\n")
    
    for which, history_dict in zip(["train", "test"], [train_loss_history, test_loss_history]):
        save_loss_history(CONFIG, history_dict, which)

        


if __name__ == '__main__':

    CONFIG = {
        'epoch': 2,
        'batch_size': 16,
        'new_vit_model': True,
        'pretrained_model_name': "google/vit-base-patch16-224", # for 'new_vit_model' = True
        'path_model_to_resume': Path(""), # for 'new_vit_model' = False
        'dataset_folder': Path("C:/Users/abcd2/Datasets/2022_icml_lens_sim/dev_256"),
        'dir_model_save': Path("./saved_model"),
        'model_file_name_prefix': 'power_law_pred_vit',
        'init_learning_rate': 1e-4,
        'record_loss_every_num_batch': 2,
        'target_keys': [
            "theta_E", 
            "gamma", 
            "center_x", 
            "center_y", 
            "e1", 
            "e2", 
            "gamma_ext", 
            "psi_ext", 
            "lens_light_R_sersic", 
            "lens_light_n_sersic",
        ]
    }

    train_model(CONFIG)



