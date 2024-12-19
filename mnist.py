import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip
import os
import struct

# Download MNIST dataset from Yann LeCun's site
def download_mnist(path="."):
    base_url = "https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    os.makedirs(path, exist_ok=True)

    for file in files:
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            print(f"Downloading {base_url}{file}...")
            urllib.request.urlretrieve(base_url + file, file_path)
        else:
            print(f"{file} already exists, skipping download.")

# Load MNIST data without sklearn
def load_mnist(path="."):
    def load_images(filename):
        with gzip.open(os.path.join(path, filename), 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols) / 255.0

    def load_labels(filename):
        with gzip.open(os.path.join(path, filename), 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_test = load_images('t10k-images-idx3-ubyte.gz')
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')

    def one_hot_encode(labels, num_classes=10):
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return X_train, X_test, y_train, y_test

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Compute loss
def compute_loss(Y, A2):
    m = Y.shape[0]
    logprobs = -np.multiply(Y.T, np.log(A2)) - np.multiply(1 - Y.T, np.log(1 - A2))
    return np.sum(logprobs) / m

# Backward propagation
def backward_propagation(X, Y, W1, b1, W2, b2, cache):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache

    dZ2 = A2 - Y.T
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Train the model
def train(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        # Forward propagation
        A2, cache = forward_propagation(X_train, W1, b1, W2, b2)

        # Compute loss
        loss = compute_loss(Y_train, A2)

        # Backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, W1, b1, W2, b2, cache)

        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    # Evaluate the model
    evaluate(X_test, Y_test, W1, b1, W2, b2)

    return W1, b1, W2, b2

# Evaluate the model
def evaluate(X, Y, W1, b1, W2, b2):
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)
    labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Visualize some predictions
def visualize_predictions(X, Y, W1, b1, W2, b2):
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)

    for i in range(5):
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Prediction: {predictions[i]}, True Label: {np.argmax(Y[i])}")
        plt.show()

# Main script
path = "mnist_data"
download_mnist(path)
X_train, X_test, Y_train, Y_test = load_mnist(path)
input_size = 784
hidden_size = 64
output_size = 10
epochs = 1000
learning_rate = 0.1

W1, b1, W2, b2 = train(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, epochs, learning_rate)
visualize_predictions(X_test, Y_test, W1, b1, W2, b2)
