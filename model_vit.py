""" Code taken from https://github.com/joshualin24/vit_strong_lensing/blob/master/vit.py 
    Was originally created by 2022-4-4 neural modelworks by Joshua Yao-Yu Lin
    Then modified by Kuan-Wei Huang
"""


import os
import datetime
import scipy.ndimage

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from transformers import ViTForImageClassification


class DeepLenstronomyDataset(Dataset):  # torch.utils.data.Dataset
    
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        # self.train_folder = 'train'#'data_train'
        # self.test_folder = 'test'#'data_test'
        self.train_folder = ''#'data_train'
        self.test_folder = ''#'data_test'

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv(self.path + '/metadata.csv')  #TODO: right csv?
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + '/metadata.csv')  #TODO: right csv?

    def __getitem__(self, index):
        img_path = self.path + self.df['img_path'].values[index][-13:]
        img = np.load(img_path)
        img = scipy.ndimage.zoom(img, 224 / 100, order=1)  #TODO: this is hard coded
        image = np.zeros((3, 224, 224))  #TODO: this is hard coded
        for i in range(3):
            image[i, :, :] += img

        target_keys = [
            "theta_E", 
            "gamma", 
            "center_x", 
            "center_y", 
            "e1", 
            "e2", 
            "source_x", 
            "source_y", 
            "gamma_ext", 
            "psi_ext", 
            "source_R_sersic", 
            "source_n_sersic", 
            "sersic_source_e1", 
            "sersic_source_e2", 
            "lens_light_e1", 
            "lens_light_e2", 
            "lens_light_R_sersic", 
            "lens_light_n_sersic",
        ]
        target_res = {key: self.df[key].iloc[[index]].values for key in target_keys}

        return image, target_res
        
    def __len__(self):
        return self.df.shape[0]


def print_n_train_params(model):
    n = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Number of trainable parameters: {n}")


def prepare_data_and_target(data, target_dict):
    data = Variable(data.float()).cuda()
    for key, val in target_dict.items():
        target_dict[key] = Variable(val.float()).cuda()
    target = torch.cat([val for _, val in target_dict.items()], dim=1)
    return data, target


def calc_avg_rms(cache):
    avg_rms = cache['total_rms'] / cache['total_counter']
    avg_rms = avg_rms.cpu()
    avg_rms = (avg_rms.data).numpy()
    cache['avg_rms'] = avg_rms
    for i in range(len(avg_rms)):
        tb.add_scalar(f"rms {i + 1}", avg_rms[i])
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


def get_train_test_datasets(dataset_folder):
    data_transform = transforms.Compose([
        transforms.ToTensor(), # scale to [0,1] and convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = torch.Tensor

    train_dataset = DeepLenstronomyDataset(
        dataset_folder, 
        train=True, 
        transform=data_transform, 
        target_transform=target_transform,
    )
    test_dataset = DeepLenstronomyDataset(
        dataset_folder, 
        train=False, 
        transform=data_transform, 
        target_transform=target_transform,
    )
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


def train_model(EPOCH, model, loss_fn, optimizer, train_loader, test_loader, 
                dir_model_save, best_test_accuracy):

    for epoch in range(EPOCH):
        model.train()
        cache_train = initialize_cache()

        # for batch_idx, (data, target_dict) in enumerate(tqdm(train_loader, total=len(train_loader))):
        for batch_idx, (data, target_dict) in enumerate(train_loader):
            data, target = prepare_data_and_target(data, target_dict)
            optimizer.zero_grad()
            output = model(data)[0] 
            loss = calc_loss(loss_fn, output, target)
            cache_train = update_cache(cache_train, output, target, loss)
            loss.backward()
            optimizer.step()

        cache_train = calc_avg_rms(cache_train)
        print_loss_rms(epoch, 'Train', cache_train)

        with torch.no_grad():
            model.eval()
            cache_test = initialize_cache()

            for batch_idx, (data, target_dict) in enumerate(test_loader):
                data, target = prepare_data_and_target(data, target_dict)
                pred = model(data)[0]
                loss = calc_loss(loss_fn, pred, target)
                cache_test = update_cache(cache_test, pred, target, loss)

                if batch_idx % test_num_batch == 0 and batch_idx != 0:
                    tb.add_scalar('test_loss', loss.item())
                    break

            cache_test = calc_avg_rms(cache_test)
            print_loss_rms(epoch, 'Test', cache_test)

            test_loss_per_batch = cache_test['total_loss'] / cache_test['total_counter']
            if test_loss_per_batch < best_test_accuracy:
                best_test_accuracy = test_loss_per_batch
                datetime_today = str(datetime.date.today())
                model_save_path = f'{dir_model_save}/power_law_pred_vit_{datetime_today}.mdl'
                torch.save(model, model_save_path)
                print(f"save model to {model_save_path}")




if __name__ == '__main__':

    EPOCH = 2
    BATCH_SIZE = 16

    test_num_batch = 50  #TODO: only for print?
    best_test_accuracy = float("inf")

    dataset_folder = Path("C:/Users/abcd2/Downloads/dev_256/")  # dir for dataset
    dir_model_save = Path("./saved_model")  # dir for saving models

    if not os.path.exists(dir_model_save):
        os.mkdir(dir_model_save)

    # LensDatasets
    train_dataset, test_dataset = get_train_test_datasets(dataset_folder)
    train_loader, test_loader = get_train_test_dataloaders(BATCH_SIZE, train_dataset, test_dataset)

    # Load Vision Transformer and modify the last layer
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    print_n_train_params(model)
    model.classifier = nn.Linear(in_features=768, out_features=18, bias=True)
    print_n_train_params(model)

    # use cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.cuda() 

    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    tb = SummaryWriter()

    #TODO: think about this. disconnected?
    # model = torch.load('./saved_model/resmodel18.mdl')  
    # print('loaded mdl!')

    train_model(EPOCH, model, loss_fn, optimizer, train_loader, test_loader, 
                dir_model_save, best_test_accuracy)

    tb.close()