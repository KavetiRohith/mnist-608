# MNIST Digit Classification Using NumPy

## Project Overview
This project implements a feed-forward neural network to classify handwritten digits (0-9) from the MNIST dataset. The implementation uses **NumPy** for numerical computations and **Matplotlib** for visualizing results. The project demonstrates how to:

- Load and preprocess the MNIST dataset.
- Train a simple neural network using backpropagation.
- Evaluate and visualize the model's predictions.

## Features
- Fully implemented neural network without external machine learning libraries.
- Cross-entropy loss function for classification tasks.
- Custom backpropagation and gradient descent implementation.
- Visualization of predictions and accuracy.

## Neural Network Architecture
- **Input Layer**: 784 neurons (28x28 pixel images).
- **Hidden Layer**: 64 neurons with sigmoid activation.
- **Output Layer**: 10 neurons (one for each digit) with sigmoid activation.

## Dataset
The MNIST dataset contains:
- 60,000 training images.
- 10,000 test images.

Each image is grayscale and normalized to a pixel range of [0, 1]. Labels are one-hot encoded.

## Requirements
- Python 3.7+
- NumPy
- Matplotlib

## Results
- **Accuracy**: ~85% on the test dataset with this very basic implementation
- **Visualization**: Example predictions and their corresponding true labels.

## Future Improvements
- Replace sigmoid with ReLU activation in hidden layers.
- Add mini-batch gradient descent.
- Experiment with deeper architectures.

## Acknowledgements
- The MNIST dataset was originally published by Yann LeCun and his collaborators.
- Inspired by basic neural network tutorials.

