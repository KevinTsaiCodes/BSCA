# BSCA (Brain Slices Classification Algorithms): A Novel Algorithm for Classifying MRI Brain Slice Images based on ResNet Technique.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKevinTsaiCodes%2FBSCA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

![training_progress.png](plotting_result%2Ftraining_progress.png)
## Introduction
BSCA (Brain Slices Classification Algorithms) is an innovative approach for accurately classifying MRI brain slice images using advanced ResNet (Residual Network) techniques. Leveraging the power of deep learning, BSCA is designed to automatically analyze and categorize MRI brain slices with exceptional precision. By employing ResNet's sophisticated architecture, BSCA enhances the model's ability to capture intricate features and patterns within the brain images, enabling improved classification accuracy. This novel algorithm holds promising potential for assisting medical professionals in diagnosing and understanding brain conditions from MRI data, contributing to more efficient and accurate healthcare assessments.

**Author**: Wei-Chun Kevin Tsai

## Requirements
### Dependencies
- Python 3.8+
- torch 2.0.1+cu118
- pydicom 2.4.2
- pillow 9.3.0
- torchvision 0.15.2+cu118
- matplotlib 3.7.1
- scikit-learn 1.2.2
- numpy 1.24.1
- opencv-contrib-python 4.8.0.74
- tqdm 4.64.1

### It was ran and tested under the following OSs:
- Windows 10 Home with GeForce GTX 3060 GPU

## Getting Started
### Usage
- Training


    python train.py --BATCH_SIZE batch_size --N_EPOCH epochs --LEARNING_RATE learning_rate
 
- To train with different settings, add **--batch_size**, **--n_epochs**, **--learing_rate** as you need.


- Testing
        
      
    python model.py --IMAGE_PATH path/to/input/image --MODEL_PATH path/to/model

- To set different settings, add **--image_path**, **--model_path** as you need.


### Demo

    python model.py --image_path testing/class1_0.jpg --model_path model/brain_slice_classifier_model.pt

### Results
![class1_0.jpg](testing%2Fclass1_0.jpg)