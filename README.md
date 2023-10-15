# BSCA (Brain Slices Classification Algorithms): A Novel Algorithm for Classifying MRI Brain Slice Images based on ResNet Technique.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKevinTsaiCodes%2FBSCA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


## IntroductionBSCA (Brain Slices Classification Algorithms) is an innovative approach designed to accurately classify MRI brain slice images using advanced ResNet (Residual Network) techniques. Harnessing the power of deep learning, BSCA aims to automatically analyze and categorize MRI brain slices with exceptional precision. By adopting ResNet's sophisticated architecture, BSCA enhances the model's ability to capture intricate features and patterns within the brain images, resulting in improved classification accuracy. This novel algorithm holds promising potential for assisting medical professionals in diagnosing and understanding brain conditions from MRI data, thus contributing to more efficient and accurate healthcare assessments in Healthy Control or Dementia.

Furthermore, BSCA is dedicated to addressing challenges that radiologists or physicians may encounter when reviewing MRI brain slice images. Traditionally, the evaluation of medical images can be time-consuming and requires specialized experts for image analysis and interpretation. BSCA's automatic classification capability provides valuable assistance to radiologists and physicians, enabling them to classify and assess a large volume of MRI brain slices more rapidly and effortlessly. This helps enhance the efficiency of medical image assessment and reduces the risk of human errors. Moreover, the high classification accuracy of BSCA also contributes to delivering more reliable diagnostic and treatment recommendations, thereby improving patient care and medical outcomes.

In summary, the application of BSCA holds the potential to advance technological developments in the medical field, providing powerful tools for medical professionals to improve brain image assessment, making it faster and more accurate, and ultimately raising the standard of patient healthcare.

**Author**: Wei-Chun Kevin Tsai

### Preparing Data
1. To build **training** dataset, you'll also need following datasets. All the images need to be **cropped into a square**, converted to **grayscale**, and resize into **512*512**.
- [ADNI 2](https://adni.loni.usc.edu/)
- [ADNI 2](https://adni.loni.usc.edu/)

2. To build **validation/testing** dataset, you'll also need following datasets. All the images need to be **cropped into a square**, converted to **grayscale**, and resize into **512*512**.
- [ADNI 2](https://adni.loni.usc.edu/)
- [ADNI 3](https://adni.loni.usc.edu/)

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
#### Training

    python train.py --BATCH_SIZE batch_size --N_EPOCH epochs --LEARNING_RATE learning_rate
 
##### To train with different settings, add `--BATCH_SIZE`, `--N_EPOCH`, `--LEARNING_RATE` as you need.


#### Testing
      
    python model_test.py --IMAGE_PATH path/to/input/image --MODEL_PATH path/to/model

##### To test with different settings, add `--IMAGE_PATH`, `--MODEL_PATH` as you need.


### Demo

    python model_test.py --image_path testing/class1_0.jpg --model_path model/brain_slice_classifier_model.pt

### Results
![class1_0.jpg](testing%2Fclass1_0.jpg)
![螢幕擷取畫面 2023-08-09 051804](https://github.com/KevinTsaiCodes/BSCA/assets/53148219/0a426b1e-4e17-44c1-a489-9f841284f8b0)
