""" Code taken from https://github.com/joshualin24/vit_strong_lensing/blob/master/vit.py 
    Was originally created by 2022-4-4 neural networks by Joshua Yao-Yu Lin
    Then modified by Kuan-Wei Huang
"""


import gc
import os
import sys
import h5py
import time
import zipfile
import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import scipy as sp
import scipy.ndimage
from scipy.ndimage import gaussian_filter, rotate

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models, utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

from transformers import ViTModel, ViTConfig, ViTFeatureExtractor, ViTForImageClassification




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
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + '/metadata.csv')  #TODO: right csv?
            #self.length = TESTING_SAMPLES

    def __getitem__(self, index):
        img_path = self.path + self.df['img_path'].values[index][-13:]
        img = np.load(img_path)
        img = scipy.ndimage.zoom(img, 224/100, order=1)  #TODO: this is hard coded
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


def calc_avg_rms(total_rms, total_counter):
    avg_rms = total_rms / total_counter
    avg_rms = avg_rms.cpu()
    avg_rms = (avg_rms.data).numpy()
    return avg_rms


def add_scalar_tb(avg_rms):
    for i in range(len(avg_rms)):
        tb.add_scalar(f"rms {i + 1}", avg_rms[i])




if __name__ == '__main__':

    EPOCH = 2
    glo_batch_size = 16
    test_num_batch = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # file name
    model_name = "gz2_hug_vit_010822B"

    # Vision Transformer
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    print_n_train_params(model)

    # # freeze the weights
    # for param in model.parameters():
    #     param.requires_grad = False

    # change the last layer
    model.classifier = nn.Linear(in_features=768, out_features=18, bias=True)
    print_n_train_params(model)

    
    # LensDatasets
    dataset_folder = Path("C:/Users/abcd2/Downloads/dev_256/")

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=glo_batch_size, 
        shuffle=True,
    )

    test_dataset = DeepLenstronomyDataset(
        dataset_folder, 
        train=False, 
        transform=data_transform, 
        target_transform=target_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=glo_batch_size, 
        shuffle=True,  #TODO shuffle for test set?
    )


    # model setup
    net = model
    net.cuda()  #TODO: combine with net = model above?
    
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr = 1e-4)
    tb = SummaryWriter()

    best_accuracy = float("inf")

    if not os.path.exists('./saved_model/'):
        os.mkdir('./saved_model/')
        # net = torch.load('./saved_model/resnet18.mdl')
        # print('loaded mdl!')

    # training
    for epoch in range(EPOCH):

        net.train()

        total_loss = 0.0
        total_counter = 0
        total_rms = 0

        # for batch_idx, (data, target_dict) in enumerate(tqdm(train_loader, total=len(train_loader))):
        for batch_idx, (data, target_dict) in enumerate(train_loader):
            data, target = prepare_data_and_target(data, target_dict)

            optimizer.zero_grad()

            output = net(data)[0]  #TODO: this is hard coded

            loss_theta_E = loss_fn(100*output[0], 100*target[0])  #TODO: this is hard coded
            loss_others = loss_fn(output, target)
            loss = loss_theta_E + loss_others

            square_diff = (output - target)
            total_rms += square_diff.std(dim=0)
            total_loss += loss.item()
            total_counter += 1

            loss.backward()
            optimizer.step()

        avg_rms = calc_avg_rms(total_rms, total_counter)
        add_scalar_tb(avg_rms)

        print(epoch, 'Train loss (averge per batch wise):', total_loss/(total_counter), 
              ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))


        with torch.no_grad():
            net.eval()

            total_loss = 0.0
            total_counter = 0
            total_rms = 0

            for batch_idx, (data, target_dict) in enumerate(test_loader):
                data, target = prepare_data_and_target(data, target_dict)

                pred = net(data)[0]  #TODO: this is hard coded

                loss_theta_E = loss_fn(100* pred[0], 100* target[0])  #TODO: this is hard coded
                loss_others = loss_fn(pred, target)
                loss = loss_theta_E + loss_others

                square_diff = (pred - target)
                total_rms += square_diff.std(dim=0)
                total_loss += loss.item()
                total_counter += 1

                if batch_idx % test_num_batch == 0 and batch_idx != 0:
                    tb.add_scalar('test_loss', loss.item())
                    break

            avg_rms = calc_avg_rms(total_rms, total_counter)
            add_scalar_tb(avg_rms)

            print(epoch, 'Test loss (averge per batch wise):', total_loss/(total_counter), 
                  ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))

            #TODO: work on the saving part later
            # if total_loss/(total_counter) < best_accuracy:
            #     best_accuracy = total_loss/(total_counter)
            #     datetime_today = str(datetime.date.today())
            #     torch.save(net, './saved_model/' + datetime_today + 'power_law_pred_vit.mdl')
            #     print("saved to " + "power_law_pred_vit.mdl" + " file.")

    tb.close()