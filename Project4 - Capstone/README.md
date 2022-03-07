# Dog Breed Prediction - Capstone Project

## Introduction

In this project, I create an app to predict dog breeds and also identify human faces in the images.

This project uses Convolution Neural Networks, CNNs, to predict the dog breeds

## Python Version and Libraries Used

    python 3.6 and above
    numpy
    random
    matplotlib
    pickle
    glob
    tqdm
    Pillow
    urllib
    sklearn
    opencV
    keras with TensorFlow backend
    jupyter notebook

## Data

   The data consists of dog images, images with human faces and other objects. It is divided into training and testing sets. The training data was supplied by Udacity, while i    borrowed the test images from https://github.com/rahulpatraiitkgp/Dog-Breed-Classifier.

## Steps taken in the development of the model
  
    Step 0: Import Datasets
    Step 1: Detect Humans
    Step 2: Detect Dogs - ResNet-50
    Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
    Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning - VGG16)
    Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning - Xception)
    Step 6: Write your Algorithm
    Step 7: Test Your Algorithm

## Model Acuracy
The CNN created from scratch had an accuracy of about 7.7%, while the VGG16 had an accuracy of 40%, while the last and the best, Xception had an accuracy of 82%.

## References
1. https://towardsdatascience.com/dog-breed-classification-using-deep-learning-concepts-23213d67936c
2. https://medium.com/nanonets/how-to-easily-build-a-dog-breed-image-classification-model-2fd214419cde
3. The Computer Vision Workshop by Hafsa Asad, Vishwesh Ravi Shrimali, and Nikhil Singh

