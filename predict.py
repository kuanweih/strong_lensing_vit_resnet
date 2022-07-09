""" Module to made prediction for test set from a given model.
"""

import torch 
import numpy as np
import pandas as pd

from tqdm import tqdm

from train_model import prepare_data_and_target, calc_pred
from src.data_utils import get_test_dataloader, get_test_dataset


class ModelPredictor:
    def __init__(self, CONFIG, path_model, file_name_pred):
        self.CONFIG = CONFIG

        test_dataset = get_test_dataset(CONFIG)
        self.test_loader = get_test_dataloader(CONFIG["batch_size"], test_dataset)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Use device = {self.device}\n")

        self.model = torch.load(path_model)
        self.model.to(self.device)
        self.model.eval()

        print("Start predicting\n")
        self.df_result = self._execute()
        self.df_result.to_csv(f"{file_name_pred}.csv", index=False)
        print("Done :)\n")

    def _execute(self):

        targets_list = self.CONFIG["target_keys_weights"].keys()

        pred_dict = {k: [] for k in targets_list}
        truth_dict = {k: [] for k in targets_list}
        sigma_dict = {k: [] for k in targets_list}

        for data, target_dict in tqdm(self.test_loader, total=len(self.test_loader)):
            
            data, _ = prepare_data_and_target(data, target_dict, self.device)
            pred_mu, pred_logvar = calc_pred(self.model, data)
            
            for ikey, key in enumerate(target_dict):
                if key in targets_list:
                    _truth = target_dict[key][:, 0].detach().tolist()
                    _pred_mu= pred_mu[:, ikey].detach().tolist()
                    _pred_logvar = pred_logvar[:,ikey].detach().tolist()
                    _sigma = np.sqrt(np.exp(_pred_logvar))

                    truth_dict[key].extend(_truth)
                    pred_dict[key].extend(_pred_mu)
                    sigma_dict[key].extend(_sigma)

        for key in targets_list:
            truth_dict[key] = np.array(truth_dict[key])
            pred_dict[key] = np.array(pred_dict[key])
            sigma_dict[key] = np.array(sigma_dict[key])

        df_truth = pd.DataFrame.from_dict(truth_dict).add_suffix('____truth')
        df_pred = pd.DataFrame.from_dict(pred_dict).add_suffix('____pred')
        df_sigma = pd.DataFrame.from_dict(sigma_dict).add_suffix('____sigma')

        df_result = pd.concat([df_truth, df_pred, df_sigma], axis=1)
        return df_result
