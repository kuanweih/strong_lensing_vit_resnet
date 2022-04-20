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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




# # vision transformer
# from transformers import ViTModel, ViTConfig, ViTFeatureExtractor, ViTForImageClassification
# #from tensorboardX import SummaryWriter


# ## file name
# model_name = "gz2_hug_vit_010822B"

# ## Vision Transformer
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# print("Number of trainable parameters: {}".format(sum(param.numel() for param in model.parameters() if param.requires_grad)))




# # freeze the weights
# #for param in model.parameters():
# #    param.requires_grad = False

# # change the last layer
# model.classifier = nn.Linear(in_features=768, out_features=18, bias=True)

# print("Number of trainable parameters: {}".format(sum(param.numel() for param in model.parameters() if param.requires_grad)))
# ### end vit





# ### LensDatasets

# folder = ""

# EPOCH = 2
# glo_batch_size = 16
# test_num_batch = 50


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# data_transform = transforms.Compose([
#             transforms.ToTensor(), # scale to [0,1] and convert to tensor
#             normalize,
#             ])
# target_transform = torch.Tensor


# class DeepLenstronomyDataset(Dataset): # torch.utils.data.Dataset
#     def __init__(self, root_dir, train=True, transform=None, target_transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train  # training set or test set
#         # self.train_folder = 'train'#'data_train'
#         # self.test_folder = 'test'#'data_test'
#         self.train_folder = 'dev_256'#'data_train'
#         self.test_folder = 'dev_256'#'data_test'
#         #self.df = pd.read_csv('../input/clean-full-train/clean_full_data.csv') #+ '/clean_full_data.csv')


#         if self.train:
#             self.path = os.path.join(self.root_dir, self.train_folder)
#             # self.df = pd.read_csv(self.path + '/lens_info.csv')
#             self.df = pd.read_csv(self.path + '/metadata.csv')


#             #self.length = TRAINING_SAMPLES
#         else:
#             self.path = os.path.join(self.root_dir, self.test_folder)
#             # self.df = pd.read_csv(self.path + '/lens_info.csv')
#             self.df = pd.read_csv(self.path + '/metadata.csv')
#             #self.length = TESTING_SAMPLES

#     def __getitem__(self, index):
#         #   gamma  center_x  center_y        e1        e2  source_x  source_y  gamma_ext  psi_ext
#         # print(self.df['name'].iloc[[index]])
#         # img_path = self.df['img_path'].iloc[[index]].values
#         img_path = 'dev_256/' + self.df['img_path'].values[index][-13:]
#         # print(img_path)
        
#         # name = self.df['name'].iloc[[index]]
        
#         theta_E = self.df['theta_E'].iloc[[index]]
#         gamma = self.df['gamma'].iloc[[index]]
#         center_x = self.df['center_x'].iloc[[index]]
#         center_y = self.df['center_y'].iloc[[index]]
#         e1 = self.df['e1'].iloc[[index]]
#         e2 = self.df['e2'].iloc[[index]]
#         source_x = self.df['source_x'].iloc[[index]]
#         source_y = self.df['source_y'].iloc[[index]]
#         gamma_ext = self.df['gamma_ext'].iloc[[index]]
#         psi_ext = self.df['psi_ext'].iloc[[index]]
#         source_R_sersic = self.df['source_R_sersic'].iloc[[index]]
#         source_n_sersic = self.df['source_n_sersic'].iloc[[index]]
#         sersic_source_e1 = self.df['sersic_source_e1'].iloc[[index]]
#         sersic_source_e2 = self.df['sersic_source_e2'].iloc[[index]]
#         lens_light_e1 = self.df['lens_light_e1'].iloc[[index]]
#         lens_light_e2 = self.df['lens_light_e2'].iloc[[index]]
#         lens_light_R_sersic = self.df['lens_light_R_sersic'].iloc[[index]]
#         lens_light_n_sersic = self.df['lens_light_n_sersic'].iloc[[index]]


#         # img_path = self.path + "/" + str(name.values[0]) + ".npy"
#         img = np.load(img_path)
#         img = scipy.ndimage.zoom(img, 224/100, order=1)
#         image = np.zeros((3, 224, 224))
#         for i in range(3):
#             image[i, :, :] += img
#         return image, theta_E.values, gamma.values, center_x.values, center_y.values, e1.values, e2.values, source_x.values, source_y.values, gamma_ext.values, psi_ext.values, source_R_sersic.values, source_n_sersic.values, sersic_source_e1.values, sersic_source_e2.values, lens_light_e1.values, lens_light_e2.values, lens_light_R_sersic.values, lens_light_n_sersic.values


#     def __len__(self):
#         return self.df.shape[0]


# train_loader = torch.utils.data.DataLoader(DeepLenstronomyDataset(folder, train=True, transform=data_transform, target_transform=target_transform),
#                     batch_size = glo_batch_size, shuffle = True
#                     )





# if __name__ == '__main__':

#     # dset_classes_number = 18
#     # net = models.resnet18(pretrained=True)
#     # num_ftrs = net.fc.in_features
#     # net.fc= nn.Linear(in_features=num_ftrs, out_features=dset_classes_number)
#     net = model
#     loss_fn = nn.MSELoss(reduction='sum')

#     net.cuda()

#     optimizer = optim.Adam(net.parameters(), lr = 1e-4)
#     tb = SummaryWriter()

#     best_accuracy = float("inf")


#     if not os.path.exists('./saved_model/'):
#         os.mkdir('./saved_model/')
#         #net = torch.load('./saved_model/resnet18.mdl')
#         #print('loaded mdl!')

#     for epoch in range(EPOCH):

#         net.train()
#         total_loss = 0.0
#         total_counter = 0
#         total_rms = 0

#         for batch_idx, (data, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic) in enumerate(tqdm(train_loader, total= len(train_loader))):
#             data, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = data.float(), theta_E.float(), gamma.float(), center_x.float(), center_y.float(), e1.float(), e2.float(), source_x.float(), source_y.float(), gamma_ext.float(), psi_ext.float(), source_R_sersic.float(), source_n_sersic.float(), sersic_source_e1.float(), sersic_source_e2.float(), lens_light_e1.float(), lens_light_e2.float(), lens_light_R_sersic.float(), lens_light_n_sersic.float()
#             data, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = Variable(data).cuda(), Variable(theta_E).cuda(), Variable(gamma).cuda(), Variable(center_x).cuda(), Variable(center_y).cuda(), Variable(e1).cuda(), Variable(e2).cuda(), Variable(source_x).cuda(), Variable(source_y).cuda(), Variable(gamma_ext).cuda(), Variable(psi_ext).cuda(), Variable(source_R_sersic).cuda(), Variable(source_n_sersic).cuda(), Variable(sersic_source_e1).cuda(), Variable(sersic_source_e2).cuda(), Variable(lens_light_e1).cuda(), Variable(lens_light_e2).cuda(), Variable(lens_light_R_sersic).cuda(), Variable(lens_light_n_sersic).cuda()

#             optimizer.zero_grad()
#             output = net(data)[0]
#             #print(output[:, 1].unsqueeze(1).shape, theta_E.shape)
#             target = torch.cat((theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic), dim = 1)
#             #target = torch.cat(target, e2)
#             #print(output.shape, target.shape)
#             #loss_theta_E = loss_fn(output[:, 0].unsqueeze(1), theta_E)
#             loss_theta_E = loss_fn(100* output[0], 100* target[0])
#             loss_others = loss_fn(output, target)
#             loss = loss_theta_E + loss_others

#             square_diff = (output - target)
#             total_rms += square_diff.std(dim=0)
#             total_loss += loss.item()
#             total_counter += 1

#             loss.backward()
#             optimizer.step()

#         # Collect RMS over each label
#         avg_rms = total_rms / (total_counter)
#         avg_rms = avg_rms.cpu()
#         avg_rms = (avg_rms.data).numpy()
#         for i in range(len(avg_rms)):
#             tb.add_scalar('rms %d' % (i+1), avg_rms[i])

#         # print test loss and tets rms
#         print(epoch, 'Train loss (averge per batch wise):', total_loss/(total_counter), ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))

#         with torch.no_grad():
#             net.eval()
#             total_loss = 0.0
#             total_counter = 0
#             total_rms = 0

#             test_loader = torch.utils.data.DataLoader(DeepLenstronomyDataset(folder, train=False, transform=data_transform, target_transform=target_transform),
#                         batch_size = glo_batch_size, shuffle = True
#                         )

#             for batch_idx, (data, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic) in enumerate(test_loader):
#                 data, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = data.float(), theta_E.float(), gamma.float(), center_x.float(), center_y.float(), e1.float(), e2.float(), source_x.float(), source_y.float(), gamma_ext.float(), psi_ext.float(), source_R_sersic.float(), source_n_sersic.float(), sersic_source_e1.float(), sersic_source_e2.float(), lens_light_e1.float(), lens_light_e2.float(), lens_light_R_sersic.float(), lens_light_n_sersic.float()
#                 data, theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic = Variable(data).cuda(), Variable(theta_E).cuda(), Variable(gamma).cuda(), Variable(center_x).cuda(), Variable(center_y).cuda(), Variable(e1).cuda(), Variable(e2).cuda(), Variable(source_x).cuda(), Variable(source_y).cuda(), Variable(gamma_ext).cuda(), Variable(psi_ext).cuda(), Variable(source_R_sersic).cuda(), Variable(source_n_sersic).cuda(), Variable(sersic_source_e1).cuda(), Variable(sersic_source_e2).cuda(), Variable(lens_light_e1).cuda(), Variable(lens_light_e2).cuda(), Variable(lens_light_R_sersic).cuda(), Variable(lens_light_n_sersic).cuda()
#                 target = torch.cat((theta_E, gamma, center_x, center_y, e1, e2, source_x, source_y, gamma_ext, psi_ext, source_R_sersic, source_n_sersic, sersic_source_e1, sersic_source_e2, lens_light_e1, lens_light_e2, lens_light_R_sersic, lens_light_n_sersic), dim = 1)
#                 #pred [batch, out_caps_num, out_caps_size, 1]
#                 pred = net(data)[0]

#                 loss_theta_E = loss_fn(100* pred[0], 100* target[0])
#                 loss_others = loss_fn(pred, target)
#                 loss = loss_theta_E + loss_others
#                 #loss = loss_fn(pred[0], target[0])


#                 square_diff = (pred - target)
#                 total_rms += square_diff.std(dim=0)
#                 total_loss += loss.item()
#                 total_counter += 1

#                 if batch_idx % test_num_batch == 0 and batch_idx != 0:
#                     tb.add_scalar('test_loss', loss.item())
#                     break

#             # Collect RMS over each label
#             avg_rms = total_rms / (total_counter)
#             avg_rms = avg_rms.cpu()
#             avg_rms = (avg_rms.data).numpy()
#             for i in range(len(avg_rms)):
#                 tb.add_scalar('rms %d' % (i+1), avg_rms[i])

#             # print test loss and tets rms
#             print(epoch, 'Test loss (averge per batch wise):', total_loss/(total_counter), ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))
#             if total_loss/(total_counter) < best_accuracy:
#                 best_accuracy = total_loss/(total_counter)
#                 datetime_today = str(datetime.date.today())
#                 torch.save(net, './saved_model/' + datetime_today + 'power_law_pred_vit.mdl')
#                 print("saved to " + "power_law_pred_vit.mdl" + " file.")

#     tb.close()