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
    def __init__(self, dir_output, dropout_on=False):
        if dropout_on:
            self.path_pred = Path(f"{dir_output}/pred_dp.csv")
        else:
            self.path_pred = Path(f"{dir_output}/pred.csv")
        self.path_config = Path(f"{dir_output}/CONFIG.npy")
        self.CONFIG = np.load(self.path_config, allow_pickle=True).item()
        self.df = pd.read_csv(self.path_pred)
        self.targets_list = self.CONFIG["target_keys_weights"].keys()

    def plot_each_pred_truth_uncertainty(self, target):

        sns.set(style="white", font_scale=1)
        fig, ax = plt.subplots(figsize=(6, 6))

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

        print(f"\nsummay stats of {target}")
        print(f"mean zscore = {zscore.mean()}")
        print(f"sigma zscore = {zscore.std()}")

        rms = np.sqrt(np.mean((pred - truth)**2))
        print(f"rms = {rms}")




class Paper2Models:
    def __init__(self, tag1, dir_output1, tag2, dir_output2, dropout_on=False):
        self.tag1 = tag1
        self.dir_output1 = dir_output1
        self.tag2 = tag2
        self.dir_output2 = dir_output2

        if dropout_on:
            self.path_pred1 = Path(f"{self.dir_output1}/pred_dp.csv")
            self.path_pred2 = Path(f"{self.dir_output2}/pred_dp.csv")
        else:
            self.path_pred1 = Path(f"{self.dir_output1}/pred.csv")
            self.path_pred2 = Path(f"{self.dir_output2}/pred.csv")

        self.path_config1 = Path(f"{self.dir_output1}/CONFIG.npy")
        self.path_config2 = Path(f"{self.dir_output2}/CONFIG.npy")
        self.CONFIG1 = np.load(self.path_config1, allow_pickle=True).item()
        self.CONFIG2 = np.load(self.path_config2, allow_pickle=True).item()

        self.dfs = {
            self.tag1: pd.read_csv(self.path_pred1),
            self.tag2: pd.read_csv(self.path_pred2),
        }

        self.targets_list = list(self.CONFIG1["target_keys_weights"].keys())
        self.n_total_sample = self.dfs[self.tag1].shape[0]

        self.dict_target_map = {
            'theta_E': r"$\theta_E$ [arcsec]",
            'gamma': r"$\gamma$",
            'center_x': r"$\theta_1$ [arcsec]",
            'center_y': r"$\theta_2$ [arcsec]",
            'e1': r"$e_1$",
            'e2': r"$e_2$",
            'lens_light_R_sersic': r"$R_{eff}$ [arcsec]",
            'lens_light_n_sersic': r"$n_{sersic}$",
        }

    def plot_random_samples(self, nsample):
        sns.set(style="white", font_scale=1)
        fig, ax = plt.subplots(2, 4, figsize=(18, 9))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        dict_alpha = {
            self.tag1: 0.5,
            self.tag2: 1,
        }
        index = np.linspace(1, self.n_total_sample - 1, nsample).astype(int)

        for tag, df in self.dfs.items():
            df_sample = df.iloc[index]

            for r in range(2):
                for c in range(4):
                    itarget = 4 * (r % 2) + c
                    target = self.targets_list[itarget]

                    truth = df_sample[f"{target}____truth"]
                    pred = df_sample[f"{target}____pred"]
                    sigma = df_sample[f"{target}____sigma"]

                    xymin = min(min(truth), min(pred))
                    xymax = max(max(truth), max(pred))

                    ax[r, c].plot([xymin, xymax], [xymin, xymax], 'k--', alpha=0.5, lw=1)
                    ax[r, c].errorbar(truth, pred, yerr=sigma, fmt='.', label=tag, 
                                      lw=1, ms=2, alpha=dict_alpha[tag])

                    ax[r, c].legend()
                    ax[r, c].set_title(self.dict_target_map[target])
                    ax[r, c].set_xlabel('truth')
                    ax[r, c].set_ylabel('prediction')

    def plot_zscore(self):

        sns.set(style="whitegrid", font_scale=1)
        fig, ax = plt.subplots(2, 4, figsize=(18, 9))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        for tag, df in self.dfs.items():

            for r in range(2):
                for c in range(4):
                    itarget = 4 * (r % 2) + c
                    target = self.targets_list[itarget]

                    truth = df[f"{target}____truth"]
                    pred = df[f"{target}____pred"]
                    sigma = df[f"{target}____sigma"]

                    zscore = (pred - truth) / sigma
        
                    ax[r, c].hist(zscore, bins=np.linspace(-5, 5, 30), 
                                  histtype='step', lw=2, label=tag)

                    ax[r, c].legend()
                    ax[r, c].set_title(self.dict_target_map[target])
                    ax[r, c].set_xlabel('(pred - truth) / sigma')
                    ax[r, c].set_ylabel('count')


    def plot_precision(self, log=False):

        sns.set(style="whitegrid", font_scale=1)
        fig, ax = plt.subplots(2, 4, figsize=(18, 9))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        for tag, df in self.dfs.items():

            for r in range(2):
                for c in range(4):
                    itarget = 4 * (r % 2) + c
                    target = self.targets_list[itarget]

                    truth = df[f"{target}____truth"]
                    pred = df[f"{target}____pred"]

                    precision = (pred - truth) / truth
        
                    ax[r, c].hist(precision, bins=np.linspace(-2, 2, 30), log=log,
                                  histtype='step', lw=2, label=tag)

                    ax[r, c].legend()
                    ax[r, c].set_title(self.dict_target_map[target])
                    ax[r, c].set_xlabel('(pred - truth) / truth')
                    ax[r, c].set_ylabel('count')


    def print_summary(self):

        
        for itarget in range(8):
            for tag, df in self.dfs.items():
                target = self.targets_list[itarget]

                truth = df[f"{target}____truth"]
                pred = df[f"{target}____pred"]
                sigma = df[f"{target}____sigma"]

                zscore = (pred - truth) / sigma
                precision = (pred - truth) / truth

                rms = np.sqrt(np.mean((pred - truth)**2))
                median_precision = np.median(precision)
                mean_precision = np.mean(precision)                
                std_precision = np.std(precision)
                median_sigma = np.median(sigma)
                mean_sigma = np.mean(sigma)

                print(f"{tag}:  {self.dict_target_map[target]}")
                print(f"    median precision = {median_precision:0.4f}")
                print(f"    mean precision = {mean_precision:0.4f}")
                print(f"    std precision = {std_precision:0.4f}")
                print(f"    median sigma = {median_sigma:0.4f}")
                print(f"    mean sigma = {mean_sigma:0.4f}")
                print(f"    rms = {rms:0.4f}")
                print(" ")
