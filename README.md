# MNIST Classification Project

A PyTorch implementation of MNIST digit classification that achieves >95% accuracy in one epoch with less than 25,000 parameters.

## Model Architecture
- Input Layer: 28x28x1
- Conv1: 8 filters with BatchNorm and ReLU
- Conv2: 16 filters with BatchNorm and ReLU
- Conv3: 20 filters with BatchNorm and ReLU
- MaxPooling layers
- Fully Connected Layer: 10 outputs
- Total Parameters: <25,000

## Key Features
- Achieves >95% accuracy in 1 epoch
- Lightweight architecture (<25K parameters)
- Uses BatchNormalization for faster convergence
- Implements dropout for regularization

## GitHub Actions Tests
The CI/CD pipeline automatically verifies:
1. Model has less than 25,000 parameters
2. Achieves accuracy greater than 95% in one epoch

## Setup and Training
1. Install dependencies: