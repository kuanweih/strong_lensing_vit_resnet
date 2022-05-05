import torch 
import numpy as np
import matplotlib.pyplot as plt

from model_vit import (
    get_train_test_dataloaders, 
    get_train_test_datasets,
    prepare_data_and_target,
)




class VisualModel:

    def __init__(self, CONFIG, model_path):

        self.show_targets = ["theta_E", "e1", "e2"]

        self.CONFIG = CONFIG
        self.model_path = model_path

        self.device = torch.device("cpu")

        self.model = torch.load(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        train_dataset, test_dataset = get_train_test_datasets(self.CONFIG)
        _, self.test_loader = get_train_test_dataloaders(self.CONFIG['batch_size'], train_dataset, test_dataset)



    def show_a_few_samples(self, n_batch, n_sample_batch):

        for batch_idx, (data, target_dict) in enumerate(self.test_loader):
            data, _ = prepare_data_and_target(data, target_dict, self.device)
            
            pred = self.model(data)[0]
            
            for isample in range(self.CONFIG['batch_size']):
                for ikey, key in enumerate(target_dict):
                    if key in self.show_targets:
                        _truth = target_dict[key][isample][0]
                        _pred = pred[isample][ikey]
                        _error= (_pred - _truth) / _truth
                        print(f"{key}: truth = {_truth: 0.4f}, pred = {_pred: 0.4f}, error = {100 * _error: 0.2f} %")
            
                plt.imshow(data[isample, 0, :, :])
                plt.show()
                
                if isample == n_sample_batch:
                    break

            if batch_idx == n_batch:
                break

    def create_complete_pred_target_dicts(self):

        # TODO: not finished yet

        self.complete_pred_dict = {k: np.zeros(test_dataset.__len__()) for k in self.show_targets}


        for batch_idx, (data, target_dict) in enumerate(self.test_loader):
            data, _ = prepare_data_and_target(data, target_dict, self.device)
            
            pred = self.model(data)[0]
            
            for isample in range(self.CONFIG['batch_size']):
                for ikey, key in enumerate(target_dict):
                    if key in self.show_targets:
        

