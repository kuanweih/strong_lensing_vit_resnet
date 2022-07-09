import os
import numpy as np




def print_n_train_params(model):
    """ Print number of trainable parameters.

    Args:
        model (model object): presumably a ViT model
    """
    n = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Number of trainable parameters: {n}")


def create_output_folder(CONFIG):
    if not os.path.exists(CONFIG['output_folder']):
        os.mkdir(CONFIG['output_folder'])
    # else:
    #     continue
        #raise ValueError(f"{CONFIG['output_folder']} already exist!")


def save_config(CONFIG):
    fname = f"{CONFIG['output_folder']}/CONFIG.npy"
    np.save(fname, CONFIG)


def list_avail_model_names():
    return [
        "google/vit-base-patch16-224",
        "resnet18",
    ]
