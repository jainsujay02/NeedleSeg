# BE 224B Course Project

## Overview

This project implements a UNet-based neural network for segmentation of needles from Chest CT images.  

### Important note: 

There are two run files: k_run.py and run.py. k_run.py contains the code for my top-performing model that I submitted on Kaggle and will recreate the exact result I submitted as my final submission on the leaderboard. Run.py is the model that I used for the paper and is designed to run with just the training image and mask dataset from Kaggle. There are a few other differences between the two models: k_run.py runs for 150 epochs, saves no logging information, and has no post-processing. run.py saves logging information to plot loss curves, has metrics for local evaluation (instead of creating the submission for Kaggle), and runs for just 20 epochs. 

All the other considerations apply equally to both files. 

## Directory Structure
Ensure the following directories exist:

- `trainImages/`: Directory containing training images in `.jpg` format.
- `trainMasks/`: Directory containing corresponding masks in `.png` format.
- `testImages/`: Directory containing test images in `.jpg` format.
- `predicted_masks/`: Directory where predicted masks are saved.
- `submissions/`: Directory where the submission file is saved.

## Update paths to the directory in code: 

In run.py, update the following variables: 

`
train_dir = 'path/to/trainImages'
mask_dir = 'path/to/trainMasks'
`

In k_run.py, update the following variables: 

`
train_dir = 'path/to/trainImages'
mask_dir = 'path/to/trainMasks'
model_path = 'path/to/model/unet_model_state.pth'
test_dir = 'path/to/testImages'
`

## Requirements
Ensure the following are installed (Conda/pip both work)

- Python 3.9 
- os (standard library)
- numpy
- pandas
- torch
- torchvision
- scikit-image
- scikit-learn
- PIL
- scipy
- matplotlib

## Usage

Select the model you would like to run in your env (recommend using Conda to create virtual env with above requirements) and execute the following cmd: 

`> python3 [filename].py`
