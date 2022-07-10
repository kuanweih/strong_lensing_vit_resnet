import torch 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from src.data_utils import get_train_test_dataloaders
from src.data_utils import get_train_test_datasets
from train_model import prepare_data_and_target, calc_pred




class VisualModel:

    def __init__(self, CONFIG, model_path):

        self.show_targets = ["theta_E","center_x", "center_y","e1", "e2"] 
        self.CONFIG = CONFIG
        self.model_path = model_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Use device = {self.device}\n")

        self.model = torch.load(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        train_dataset, test_dataset = get_train_test_datasets(self.CONFIG)
        _, self.test_loader = get_train_test_dataloaders(self.CONFIG['batch_size'], train_dataset, test_dataset)
        
        self.n_test = test_dataset.__len__()
        
        self.pred_dict, self.truth_dict, self.sigma_dict = self.get_pred_truth_dicts()


    def show_a_few_samples(self, n_batch, n_sample_batch):

        for batch_idx, (data, target_dict) in enumerate(self.test_loader):
            if batch_idx == n_batch:
                break
                
            data, _ = prepare_data_and_target(data, target_dict, self.device)
            pred = self.model(data)[0] 
            
            for isample in range(self.CONFIG['batch_size']):
                if isample == n_sample_batch:
                    break
                
                for ikey, key in enumerate(target_dict):
                    if key in self.show_targets:
                        _truth = target_dict[key][isample][0].data
                        _pred = pred[isample][ikey].data
                        _error= (_pred - _truth) / _truth
                        print(f"{key}: truth = {_truth: 0.4f}, pred = {_pred: 0.4f}, error = {100 * _error: 0.2f} %")
            
                plt.imshow(data.cpu()[isample, 0, :, :])
                plt.show()
    
    def show_a_few_samples_uncertainty(self, n_batch, n_sample_batch):

        for batch_idx, (data, target_dict) in enumerate(self.test_loader):
            if batch_idx == n_batch:
                break
                
            data, _ = prepare_data_and_target(data, target_dict, self.device)
            pred_mu, pred_logvar = calc_pred(self.model, data)
            
            for isample in range(self.CONFIG['batch_size']):
                if isample == n_sample_batch:
                    break
                
                for ikey, key in enumerate(target_dict):
                    if key in self.show_targets:
                        _truth = target_dict[key][isample][0].data.cpu()
                        _pred_mu = pred_mu[isample][ikey].data.cpu()
                        _pred_logvar = pred_logvar[isample][ikey].data.cpu()
                        _sigma = np.sqrt(np.exp(_pred_logvar))                        
                        _chi= np.sqrt((_pred_mu - _truth)**2 / _sigma**2)
                        print(f"{key}: truth = {_truth: 0.4f}, pred = {_pred_mu: 0.4f}, sigma = {_sigma: 0.2f}")
            
                plt.imshow(data.cpu()[isample, 0, :, :])
                plt.show()


    def get_pred_truth_dicts(self):

        pred_dict = {k: [] for k in self.show_targets}
        truth_dict = {k: [] for k in self.show_targets}
        sigma_dict = {k: [] for k in self.show_targets}

        for batch_idx, (data, target_dict) in enumerate(tqdm(self.test_loader, total=len(self.test_loader))):

             #TODO: remove this
            #if batch_idx == 4:
            #    break
            
            data, _ = prepare_data_and_target(data, target_dict, self.device)
            #pred = self.model(data)[0]
            pred_mu, pred_logvar = calc_pred(self.model, data)
            
            for ikey, key in enumerate(target_dict):
                if key in self.show_targets:
                    _truth = target_dict[key][:, 0].detach().tolist()
                    _pred_mu= pred_mu[:, ikey].detach().tolist()
                    _pred_logvar = pred_logvar[:,ikey].detach().tolist()
                    _sigma = np.sqrt(np.exp(_pred_logvar))

                    truth_dict[key].extend(_truth)
                    pred_dict[key].extend(_pred_mu)
                    sigma_dict[key].extend(_sigma)

        for key in self.show_targets:
            truth_dict[key] = np.array(truth_dict[key])
            pred_dict[key] = np.array(pred_dict[key])
            sigma_dict[key] = np.array(sigma_dict[key])

        return pred_dict, truth_dict, sigma_dict


    def plot_each_pred_truth(self, target_key):
        
        sns.set(style="white", font_scale=1)
        fig, ax = plt.subplots()

        ax.set_aspect('equal', adjustable='box')
        
        x = self.truth_dict[target_key]
        y = self.pred_dict[target_key]

        xymin = min(min(x), min(y))
        xymax = max(max(x), max(y))

        ax.hexbin(x, y, extent=(xymin, xymax, xymin, xymax))
        ax.plot([xymin, xymax], [xymin, xymax], 'w--', alpha=0.5)

        ax.set_title(target_key)
        ax.set_xlabel('truth')
        ax.set_ylabel('prediction')
        
    def plot_each_pred_truth_uncertainty(self, target_key):
        
        sns.set(style="white", font_scale=1)
        fig, ax = plt.subplots(figsize=(8,8))
        plt.figure(figsize=(3, 3))

        ax.set_aspect('equal', adjustable='box')
        
        x = self.truth_dict[target_key]
        y = self.pred_dict[target_key]
        z = self.sigma_dict[target_key]

        xymin = min(min(x), min(y))
        xymax = max(max(x), max(y))

        ax.hexbin(x, y, extent=(xymin, xymax, xymin, xymax))
        ax.plot([xymin, xymax], [xymin, xymax], 'w--', alpha=0.5)
        
        #pick random 20 points to show the error bars
        index = np.linspace(0, len(x)-1, 20).astype(int)
        x_select = [x[i] for i in index]
        y_select = [y[i] for i in index]
        z_select = [z[i] for i in index]
        ax.errorbar(x_select, y_select, yerr=z_select, fmt='o')
        
        ax.set_title(target_key)
        ax.set_xlabel('truth')
        ax.set_ylabel('prediction')

        


class Visual_loss:
    
    def __init__(self, CONFIG):
        
        self.dir = CONFIG['output_folder']
    
    def plot_train_test_loss(self):
        
        test_history_path = f"{self.dir}/test_history.npy"
        test_history= np.load(test_history_path, allow_pickle=True)
        
        test_mse = test_history.item().get('mse')
        test_epoch= test_history.item().get('epoch')
        
        train_history_path = f"{self.dir}/train_history.npy"
        train_history= np.load(train_history_path, allow_pickle=True)
        
        train_mse = train_history.item().get('mse')
        train_epoch = train_history.item().get('epoch')
        
        fig, ax = plt.subplots()
        #ax.set_aspect('equal', adjustable='box')
        
        ax.plot(test_mse, 'b-', label='test')
        ax.plot(train_mse, 'r--', label='train')
        
        ax.legend()
        ax.set_title("loss")
        ax.set_xlabel('epoch')
        ax.set_ylabel('mse')




class PredVisualizer:
    def __init__(self, dir_output):
        self.path_pred = Path(f"{dir_output}/pred.csv")
        self.path_config = Path(f"{dir_output}/CONFIG.npy")
        self.CONFIG = np.load(self.path_config, allow_pickle=True).item()
        self.df = pd.read_csv(self.path_pred)
        self.targets_list = self.CONFIG["target_keys_weights"].keys()

    def plot_each_pred_truth_uncertainty(self, target):

        sns.set(style="white", font_scale=1)
        fig, ax = plt.subplots(figsize=(6, 6))
        # plt.figure(figsize=(3, 3))

        ax.set_aspect('equal', adjustable='box')
        x = self.df[f"{target}____truth"]
        y = self.df[f"{target}____pred"]
        z = self.df[f"{target}____sigma"]

        xymin = min(min(x), min(y))
        xymax = max(max(x), max(y))

        ax.hexbin(x, y, extent=(xymin, xymax, xymin, xymax))
        ax.plot([xymin, xymax], [xymin, xymax], 'w--', alpha=0.5)

        #pick random 20 points to show the error bars
        index = np.linspace(0, len(x)-1, 20).astype(int)
        x_select = [x[i] for i in index]
        y_select = [y[i] for i in index]
        z_select = [z[i] for i in index]
        ax.errorbar(x_select, y_select, yerr=z_select, fmt='.')

        ax.set_title(target)
        ax.set_xlabel('truth')
        ax.set_ylabel('prediction')

    def plot_each_zscore(self, target):

        sns.set(style="white", font_scale=1)
        fig, ax = plt.subplots(figsize=(4, 4))

        truth = self.df[f"{target}____truth"]
        pred = self.df[f"{target}____pred"]
        sigma = self.df[f"{target}____sigma"]
        
        zscore = (pred - truth) / sigma
        
        ax.hist(zscore, histtype='step', lw=2)

        ax.set_title(target)
        ax.set_xlabel('(pred - truth) / sigma')
        ax.set_ylabel('count')

        print(f"mean zscore = {zscore.mean()}")
        print(f"sigma zscore = {zscore.std()}")

        mse = np.mean((pred - truth)**2)
        print(f"mse = {mse}")
