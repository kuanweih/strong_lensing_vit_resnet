# Strong Gravitational Lensing Parameter Estimation with Vision Transformer

This repo contains all our code for using the Vision Transformer models on the imaging multi-regression task of parameter and uncertainty estimation for strong lensing systems. The paper is published in the [ECCV 2022 Workshops](https://link.springer.com/chapter/10.1007/978-3-031-25056-9_10) and this is [the arXiv link](https://arxiv.org/pdf/2210.04143.pdf).

Authors: Kuan-Wei Huang, Geoff Chih-Fan Chen, Po-Wen Chang, Sheng-Chieh Lin, ChiaJung Hsu, Vishal Thengane, and Joshua Yao-Yu Lin

## Data generation / preparation using Lenstronomy
- [This notebook](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/notebooks/Lenstronomy_simulation_dev.ipynb) is used to generate the images (data) and paramters (targets) as the dataset for the strong lensing systems.
- [This notebook](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/notebooks/split_geoff_30000.ipynb) is used to process the dataset: data split and target normalization. 

## Source code for training models
- [`train_model.py`](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/train_model.py) is the main code to train models (ViT and ResNet).
- [The `src` folder](https://github.com/kuanweih/strong_lensing_vit_resnet/tree/main/src) contains scripts for helper functions. 

## Train models
- [This notebook](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/notebooks/training_eccv/train_vit_geoff_30000_vit_new_2.ipynb) is used to train a ViT model.
- [The `training_eccv` folder](https://github.com/kuanweih/strong_lensing_vit_resnet/tree/main/notebooks/training_eccv) contains the notebooks used to train models for our ECCV paper.
- [The `training_icml` folder](https://github.com/kuanweih/strong_lensing_vit_resnet/tree/main/notebooks/training_icml) contains the notebooks used to train models for our ICML paper.

## Model Prediction and visulization
- [`predict.py`](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/predict.py) is the source code to make prediction using a trained model. 
- [This notebook](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/notebooks/pred_eval/pred_eccv.ipynb) uses `predict.py` to make predictions for our ECCV paper.
- [`visualization.py`](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/visualization.py) contains objects and functions for visulization.
- [This notebook](https://github.com/kuanweih/strong_lensing_vit_resnet/blob/main/notebooks/pred_eval/eval_viz_eccv.ipynb) uses `visualization.py` to make figures for our ECCV paper.
