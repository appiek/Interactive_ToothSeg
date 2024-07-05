## Overview
We present a novel and efficient interactive computing framework for segmenting the alveolar bone and individual teeth from CBCT scans by combining a deep convolutional neural network with a proposed interactive tooth landmark annotation approach. We implemented our method based on the open-source machine learning framework Pytorch. 

## Demo
![Example Image1](https://github.com/appiek/Interactive_CNN_based_CBCT_Segmentation_Pytorch_Demo/blob/main/Demo1.png)
![Example Image2](https://github.com/appiek/Interactive_CNN_based_CBCT_Segmentation_Pytorch_Demo/blob/main/Demo2.png)


## Dependencies
* Python 3.8 or higher version
* Pytorch 2.0.0
* Medpy 0.4.0
* PyQT5 5.15.10
* VTK 9.0.0
* Scikit-image 17.2
* opencv-python 4.5.3.56
* Numpy
* Scipy

## Composition of code
1. main.py: Construct the visual interface based on PyQT5 and Interactive tooth landmark annotation tools.
2. /nets: model construction
3. Tooth_Alveolar_Construction.py: Implementation of the alveolar bone and segmentation.

## Quick Start
* Testing: if you just want to validate the segmentation performance of pre-trained models, follow these steps:
   1. Download our code on your computer, assume the path is "./";
   2. Download the testing data [Link](https://pan.baidu.com/s/1nIrYfkmogeZHI0NCFZRxHw?pwd=1234) and unzip this file in your computer
   3. Download the pre-trained parameters of model [Link](https://pan.baidu.com/s/1BhCx5SayUGWYTipsPQl-AA?pwd=1234) and unzip this file into the path './checkpoints/'
   4. Run 'main.py' for testiing the performance of method
   5. We also provide an EXE version [Link](https://pan.baidu.com/s/1Hzh2eHhx1SX6ulNakvY2Zw?pwd=1234) to test our method. 

## Contact information  
* E-mail: xlpflyinsky@foxmail.com
