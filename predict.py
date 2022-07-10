""" Module to made prediction for test set from a given model.
"""

import torch 
import numpy as np
import pandas as pd

from pathlib import Path
from pickle import load
from tqdm import tqdm

from train_model import prepare_data_and_target, calc_pred
from src.data_utils import get_test_dataloader, get_test_dataset


class ModelPredictor:
    def __init__(self, dir_output, path_model,
                 dataset_folder=Path("C:/Users/abcd2/Datasets/2022_icml_lens_sim/geoff_1200")):
        self.dir_output = dir_output                 
        self.path_model = path_model
        self.dataset_folder = dataset_folder

        path_config = Path(f"{dir_output}/CONFIG.npy")
        self.CONFIG = np.load(path_config, allow_pickle=True).item()
        self.CONFIG["dataset_folder"] = self.dataset_folder
        self.CONFIG["batch_size"] = 10
        print(self.CONFIG)

        test_dataset = get_test_dataset(self.CONFIG)
        self.test_loader = get_test_dataloader(self.CONFIG["batch_size"], test_dataset)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Use device = {self.device}\n")

        self.model = torch.load(path_model)
        self.model.to(self.device)
        self.model.eval()
        
        self.targets_list = self.CONFIG["target_keys_weights"].keys()

    def execute(self):
        print("Start predicting\n")

        pred_dict = {k: [] for k in self.targets_list}
        truth_dict = {k: [] for k in self.targets_list}
        sigma_dict = {k: [] for k in self.targets_list}

        for data, target_dict in tqdm(self.test_loader, total=len(self.test_loader)):
            
            data, _ = prepare_data_and_target(data, target_dict, self.device)
            pred_mu, pred_logvar = calc_pred(self.model, data)
            
            for ikey, key in enumerate(target_dict):
                if key in self.targets_list:
                    _truth = target_dict[key][:, 0].detach().tolist()
                    _pred_mu= pred_mu[:, ikey].detach().tolist()
                    _pred_logvar = pred_logvar[:,ikey].detach().tolist()
                    _sigma = np.sqrt(np.exp(_pred_logvar))

                    truth_dict[key].extend(_truth)
                    pred_dict[key].extend(_pred_mu)
                    sigma_dict[key].extend(_sigma)

        for key in self.targets_list:
            truth_dict[key] = np.array(truth_dict[key])
            pred_dict[key] = np.array(pred_dict[key])
            sigma_dict[key] = np.array(sigma_dict[key])

        df_truth = pd.DataFrame.from_dict(truth_dict).add_suffix('____truth')
        df_pred = pd.DataFrame.from_dict(pred_dict).add_suffix('____pred')
        df_sigma = pd.DataFrame.from_dict(sigma_dict).add_suffix('____sigma')

        self.df_result = pd.concat([df_truth, df_pred, df_sigma], axis=1)

        path_pred_scaled = Path(f"{self.dir_output}/pred_scaled.csv")
        self.df_result.to_csv(path_pred_scaled, index=False)
        print(f"Saved pred_scaled.csv to {path_pred_scaled} \n")

    def scale_back(self, path_scaler=Path("C:/Users/abcd2/Datasets/2022_icml_lens_sim/geoff_30000/scaler.pkl")):
        print("Start scaling pred_scaled.csv back\n")

        scaler = load(open(path_scaler, 'rb'))
        df_resumed = {}

        for suffix in ["truth", "pred", "sigma"]:
            for target in self.targets_list:
                key = f"{target}____{suffix}"
                
                mask = scaler.feature_names_in_ == target
                mu = scaler.mean_[mask][0]
                std = scaler.scale_[mask][0]

                df_resumed[key] = mu + self.df_result[key] * std

        self.df_resumed = pd.DataFrame(df_resumed)

        path_pred = Path(f"{self.dir_output}/pred.csv")
        self.df_resumed.to_csv(path_pred, index=False)
        print(f"Scaled back and saved pred.csv to {path_pred} \n")

        # # sanity check
        # df_meta = pd.read_csv(f"{self.dataset_folder}/metadata.csv")
        # for target in self.targets_list:
        #     key = f"{target}____{suffix}"
        #     print(np.mean((self.df_resumed[key] - df_meta[target])**2))
        

                
                
                



