# BreedBot


This repository contains a Dog Breed Classification system built using MobileNet, a pre-trained convolutional neural network model available from TensorFlow Hub. The system can classify images of dogs into various breeds.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

The Dog Breed Classification System utilizes MobileNet, a lightweight convolutional neural network, to classify images of dogs into their respective breeds. This project demonstrates how to leverage TensorFlow Hub's pre-trained models for transfer learning.

## Features

- Efficient classification of dog breeds using MobileNet.
- Pre-trained model from TensorFlow Hub ensures high accuracy and fast inference.
- Easy-to-use script for training and testing the model.

## Requirements

- Python 3.6+
- TensorFlow
- TensorFlow Hub
- NumPy
- OpenCV
- Matplotlib

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/dog-breed-classification.git
    cd dog-breed-classification
    ```

2. Install the required libraries:

    ```sh
    pip install tensorflow tensorflow-hub numpy opencv-python matplotlib
    ```



## Usage

1. **Prepare your dataset**: Ensure you have a dataset of dog images, organized by breed.

2. **Define model parameters**:

    Before building the model, define the input and output shapes:
    
    ```python
    INPUT_SHAPE = (224, 224, 3)  # Input shape for MobileNet
    OUTPUT_SHAPE = (NUMBER_OF_BREEDS,)  # Number of dog breeds to classify
    MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"  # URL of the pre-trained MobileNet model
    ```

3. **Build and train the model**.

4. **Evaluate the model**.

5. **Make predictions**.

## Model Details

The model used in this project is MobileNetV2, a lightweight and efficient convolutional neural network designed for mobile and embedded vision applications. The specific model used is available from TensorFlow Hub:

- **URL**: [https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4)
- **Input Shape**: (224, 224, 3)
- **Output Shape**: Number of dog breeds to classify

## Dataset

The dataset used for this project is from the Kaggle Dog Breed Identification competition. It contains a large number of images of dogs, labeled by breed.

- **Kaggle Dataset**: [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)

## Project Structure

```
dog-breed-classification/
├── model.py                 # Script for building and training the model and makeing prediction
├── README.md                # Project README file
```

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) and [TensorFlow Hub](https://www.tensorflow.org/hub) for providing the frameworks and pre-trained models.
- The creators of MobileNet for their efficient and high-performance model design.
- [Kaggle](https://www.kaggle.com/) for providing the Dog Breed Identification dataset.

