""" Code taken from https://github.com/joshualin24/vit_strong_lensing/blob/master/vit.py 
    Originally created by 2022-4-4 neural modelworks by Joshua Yao-Yu Lin
    Modified by Kuan-Wei Huang
"""

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from transformers import ViTForImageClassification

from src.data_utils import (
    get_train_test_datasets,
    get_train_test_dataloaders,
)

from src.cache_utils import (
    CacheEpoch,
    CacheHistory,
)

from src.helpers import (
    print_n_train_params,
    create_output_folder,
    save_config,
)




def prepare_data_and_target(data, target_dict, device):
    """ Prepare data (X) and target (Y) for a given batch.

    Args:
        data (torch.Tensor): image (X)
        target_dict (dict): targets (Y)
        device (torch.device): cpu or gpu

    Returns:
        data (torch.Tensor): image (X)
        target (torch.Tensor): targets(Y)
    """
    data = data.float().to(device)
    for key, val in target_dict.items():
        target_dict[key] = val.float().to(device)
    target = torch.cat([val for _, val in target_dict.items()], dim=1)
    return data, target


def nll_diagonal(target, mu, logvar, device, CONFIG):
    """Evaluate the NLL for single Gaussian with diagonal covariance matrix
    Parameters
    ----------
    target : torch.Tensor of shape [batch_size, Y_dim]
        Y labels
    mu : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the mu (mean parameter) of the BNN posterior
    logvar : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the log of the diagonal elements of the covariance matrix
    Returns
    -------
    torch.Tensor of shape
        NLL values
    """
    weight = [w for _, w in CONFIG["target_keys_weights"].items()]
    weight = torch.tensor(weight, requires_grad=False).to(device)
    weight = weight / torch.sum(weight)

    precision = torch.exp(-logvar)
    sq_err = (target - mu) * (target - mu)

    loss = 0.5 * (precision*sq_err + logvar + np.log(2*np.pi))
    loss = torch.sum(loss*weight, dim=1)  # weighted sum accross targets 
    loss = torch.mean(loss, dim=0) # accross batch samples
    return loss


def load_model(CONFIG):
    """ Load a model based on CONFIG.

    Args:
        CONFIG (dict): CONFIG

    Raises:
        ValueError: CONFIG['new_model_name'] needs to match.

    Returns:
        model: model ready for training.
    """
    if CONFIG['load_new_model']:
        model_name = CONFIG['new_model_name']
        n_targets = len(CONFIG['target_keys_weights']) 
        out_features = 2 * n_targets  # double the len for uncertainties

        if model_name == "google/vit-base-patch16-224":
            model = ViTForImageClassification.from_pretrained(
                model_name, 
                hidden_dropout_prob=CONFIG['dropout_rate'],
                attention_probs_dropout_prob=CONFIG['vit_attention_dropout_rate'],
            )
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(in_features=num_ftrs, out_features=out_features, bias=True)
        elif model_name.startswith("resnet"):
            model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(in_features=num_ftrs, out_features=out_features, bias=True)
            if CONFIG['dropout_rate'] > 0:
                append_dropout(model, CONFIG['dropout_rate'])
        else:
            raise ValueError(f"{model_name} not a valid model name!")

        print(f"Use fresh pretrained model = {CONFIG['new_model_name']}\n")
        print_n_train_params(model)
        print(" ")
    else:
        model = torch.load(CONFIG['resumed_model_path'])  
        print(f"Use our trained model = {CONFIG['resumed_model_path']}\n")
    return model


def append_dropout(model, dropout_rate):
    """ Append dropout layer after each ReLU layer.

    Args:
        model (pytorch model object): model for adding dropouts.
        dropout_rate (float, optional): dropout rate.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, dropout_rate)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=dropout_rate))
            setattr(model, name, new)


def save_model(CONFIG, model, epoch, test_loss):
    """ Save trained model at the end of an epoch.

    Args:
        CONFIG (dict): CONFIG
        model (pytorch model object): model to be saved
        epoch (int): epoch
        test_loss (float): test loss of the model
    """
    _dir = CONFIG['output_folder']
    model_save_path = f"{_dir}/epoch_{epoch}_testloss_{test_loss:.6f}.mdl"
    torch.save(model, model_save_path)
    print(f"\nSave model to {model_save_path}\n")


def calc_pred(model, data):
    """ Calculate prediction of input data using model.

        pred = model(data)

    Different model objects have different pred shapes by default such as 
    ViT and ResNet and that's what the if statements are for.

        'pred' will be split into 'pred_mu' and 'pred_logvar'

    Args:
        model (model object): ViT or ResNet
        data (torch.Tensor): batch data parsed in the model

    Raises:
        TypeError: type(model) has to be checked

    Returns:
        [torch.Tensor]: pred_mu: target prediction 
        [torch.Tensor]: pred_logvar: prediction log variance
    """
    if isinstance(model, ViTForImageClassification):
        pred = model(data)[0]
    elif isinstance(model, torchvision.models.resnet.ResNet):
        pred = model(data)
    else:
        raise TypeError(f"{type(model)} not implemented for correct pred shape.")
    
    pred_mu, pred_logvar = torch.tensor_split(pred, 2, dim=1)

    return pred_mu, pred_logvar


def train_model(CONFIG):
    """ Train models based on parameters in CONFIG.

    Args:
        CONFIG (dict): model configuration dict
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Use device = {device}\n")

    create_output_folder(CONFIG)
    save_config(CONFIG)

    # prepare data loaders
    train_dataset, test_dataset = get_train_test_datasets(CONFIG)
    train_loader, test_loader = get_train_test_dataloaders(CONFIG['batch_size'], train_dataset, test_dataset)

    # load model and cast to 'device'
    model = load_model(CONFIG)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['init_learning_rate'])

    best_test_loss = float("inf")  # to identify best model ever seen
    
    train_history = CacheHistory("train")
    test_history = CacheHistory("test")


    for epoch in range(CONFIG['epoch']):

        model.train()
        cache_train = CacheEpoch("train")

        for data, target_dict in tqdm(train_loader, total=len(train_loader)):
            data, target = prepare_data_and_target(data, target_dict, device)
            optimizer.zero_grad()
            pred_mu, pred_logvar = calc_pred(model, data)
            loss = nll_diagonal(target, pred_mu, pred_logvar, device, CONFIG)

            cache_train.update_cache(pred_mu, target)

            loss.backward()
            optimizer.step()

        cache_train.calc_avg_across_batches()
        cache_train.print_cache(epoch)


        with torch.no_grad():
            model.eval()
            cache_test = CacheEpoch("test")

            for data, target_dict in test_loader:
                data, target = prepare_data_and_target(data, target_dict, device)
                pred_mu, pred_logvar = calc_pred(model, data)  
                loss = nll_diagonal(target, pred_mu, pred_logvar, device, CONFIG)

                cache_test.update_cache(pred_mu, target)

            cache_test.calc_avg_across_batches()
            cache_test.print_cache(epoch)

            # save model with best test loss so far
            if cache_test.avg_mse < best_test_loss:
                best_test_loss = cache_test.avg_mse
                save_model(CONFIG, model, epoch, cache_test.avg_mse)

            train_history.record_and_save(epoch, cache_train, CONFIG)
            test_history.record_and_save(epoch, cache_test, CONFIG)




if __name__ == '__main__':

    from src.helpers import list_avail_model_names

    print(list_avail_model_names())

    CONFIG = {
        "epoch": 10,
        "batch_size": 30,
        "load_new_model": True,
        'new_model_name': "google/vit-base-patch16-224",  # for 'load_new_model' = True
        # "new_model_name": "resnet152",  # for 'load_new_model' = True
        "resumed_model_path": Path(""),  # for 'load_new_model' = False
        "output_folder": Path("C:/Users/abcd2/Downloads/tmp_dev_outputs"),  # needs to be non-existing
        "dataset_folder": Path("C:/Users/abcd2/Datasets/2022_icml_lens_sim/dev_256"),
        # 'dataset_folder': Path("C:/Users/abcd2/Datasets/2022_icml_lens_sim/geoff_30000"),
        "init_learning_rate": 1e-3,
        "dropout_rate": 0.1,
        "vit_attention_dropout_rate": 0.0,  # optional for vit models
        "target_keys_weights": {
            "theta_E": 1, 
            "gamma": 1, 
            "center_x": 1, 
            "center_y": 1, 
            "e1": 1, 
            "e2": 1, 
            "lens_light_R_sersic": 1, 
            "lens_light_n_sersic": 1,
            # "gamma_ext": 1, 
            # "psi_ext": 1, 
        }
    }

    train_model(CONFIG)



