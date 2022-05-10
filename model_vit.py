""" Code taken from https://github.com/joshualin24/vit_strong_lensing/blob/master/vit.py 
    Originally created by 2022-4-4 neural modelworks by Joshua Yao-Yu Lin
    Modified by Kuan-Wei Huang
"""


from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
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
        data (np.array): image (X)
        target_dict (dict): targets (Y)
        device (torch.device): cpu or gpu

    Returns:
        data (torch.Tensor): image (X)
        target (torch.Tensor): targets(Y)
    """
    data = Variable(data.float()).to(device)
    for key, val in target_dict.items():
        target_dict[key] = Variable(val.float()).to(device)
    target = torch.cat([val for _, val in target_dict.items()], dim=1)
    return data, target


def calc_loss(pred, target, CONFIG, device):
    """ Average weighted mean squared loss per sample.

    Let 
        i = range(0, B): sample index in a batch with size B
        j = range(0, T): target index for T targets
        pred_(i, j): prediction for target j and sample i
        truth_(i, j): target for target j and sample i
        SE_(i, j): squared error for target j and sample i
        MSE_j: mean squared error for target j per sample
        w_j: weight for target j
        W = sum(w_j for j in range(T)): total weight

    SE_(i, j) = (pred_(i, j) - truth_(i, j))**2
    MSE_j = sum(SE_(i, j) for i in range(B)) / B
    Loss = sum(w_j * MSE_j for j in range(T)) / W
    
    Args:
        pred (torch.Tensor): prediction of a batch
        target (torch.Tensor): target of a batch
        CONFIG (dict): CONFIG
        device (torch.device): cpu or gpu

    Returns:
        [torch.Tensor]: Loss
    """
    weight = [w for _, w in CONFIG["target_keys_weights"].items()]
    weight = torch.tensor(weight, requires_grad=False).to(device)
    loss = torch.mean((pred - target)**2, axis=0)
    loss = torch.sum(weight * loss) / weight.sum()
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

        if model_name == "google/vit-base-patch16-224":
            model = ViTForImageClassification.from_pretrained(model_name)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(in_features=num_ftrs, out_features=n_targets, bias=True)
        elif model_name == "resnet18":
            model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(in_features=num_ftrs, out_features=n_targets, bias=True)
        else:
            raise ValueError(f"{model_name} not a valid model name!")

        print(f"Use fresh pretrained model = {CONFIG['new_model_name']}\n")
        print_n_train_params(model)
        print(" ")
    else:
        model = torch.load(CONFIG['resumed_model_path'])  
        print(f"Use our trained model = {CONFIG['resumed_model_path']}\n")
    return model


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
            output = model(data)[0]
            loss = calc_loss(output, target, CONFIG, device)

            cache_train.update_cache(output, target, loss)

            loss.backward()
            optimizer.step()

        cache_train.calc_avg_across_batches()
        cache_train.print_cache(epoch)


        with torch.no_grad():
            model.eval()
            cache_test = CacheEpoch("test")

            for data, target_dict in test_loader:
                data, target = prepare_data_and_target(data, target_dict, device)
                pred = model(data)[0]
                loss = calc_loss(pred, target, CONFIG, device)

                cache_test.update_cache(pred, target, loss)

            cache_test.calc_avg_across_batches()
            cache_test.print_cache(epoch)

            # save model with best test loss so far
            if cache_test.avg_loss < best_test_loss:
                best_test_loss = cache_test.avg_loss
                save_model(CONFIG, model, epoch, cache_test.avg_loss)

            train_history.record_and_save(epoch, cache_train, CONFIG)
            test_history.record_and_save(epoch, cache_test, CONFIG)




if __name__ == '__main__':

    from src.helpers import list_avail_model_names

    print(list_avail_model_names())

    CONFIG = {
        'epoch': 4,
        'batch_size': 30,
        'load_new_model': True,
        'new_model_name': "google/vit-base-patch16-224",  # for 'load_new_model' = True
        # 'new_model_name': "resnet18",  # for 'load_new_model' = True
        'resumed_model_path': Path(""),  # for 'load_new_model' = False
        'output_folder': Path("C:/Users/abcd2/Downloads/tmp_dev_outputs"),  # needs to be non-existing
        'dataset_folder': Path("C:/Users/abcd2/Datasets/2022_icml_lens_sim/dev_256"),
        'init_learning_rate': 1e-3,
        'target_keys_weights': {
            "theta_E": 10, 
            "gamma": 1, 
            "center_x": 1, 
            "center_y": 1, 
            "e1": 1, 
            "e2": 1, 
            "gamma_ext": 1, 
            "psi_ext": 1, 
            "lens_light_R_sersic": 1, 
            "lens_light_n_sersic": 1,
        }
    }

    train_model(CONFIG)



