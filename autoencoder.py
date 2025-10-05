# Step 3: Simple Autoencoder (from scratch, no libraries)
# This script trains a multi-layer perceptron autoencoder on the spatiotemporal matrix.

import pickle
import random
import math

# Utility functions

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def mse_loss(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

# Simple MLP Autoencoder
class Autoencoder:
    def __init__(self, input_size, hidden_size):
        # Initialize weights
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Encoder weights
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        # Decoder weights
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b2 = [0.0 for _ in range(input_size)]

    def encode(self, x):
        # x: input vector
        h = [0.0 for _ in range(self.hidden_size)]
        for i in range(self.hidden_size):
            h[i] = sigmoid(sum(self.W1[i][j] * x[j] for j in range(self.input_size)) + self.b1[i])
        return h

    def decode(self, h):
        y = [0.0 for _ in range(self.input_size)]
        for i in range(self.input_size):
            y[i] = sigmoid(sum(self.W2[i][j] * h[j] for j in range(self.hidden_size)) + self.b2[i])
        return y

    def forward(self, x):
        h = self.encode(x)
        y = self.decode(h)
        return h, y

    def train(self, data, epochs=10, lr=0.01):
        for epoch in range(epochs):
            total_loss = 0.0
            for x in data:
                # Forward pass
                h = self.encode(x)
                y = self.decode(h)
                # Compute loss
                loss = mse_loss(x, y)
                total_loss += loss
                # Backpropagation (manual, simple)
                # Output layer gradients
                dL_dy = [(y[i] - x[i]) * dsigmoid(y[i]) for i in range(self.input_size)]
                # Decoder weights update
                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        self.W2[i][j] -= lr * dL_dy[i] * h[j]
                    self.b2[i] -= lr * dL_dy[i]
                # Hidden layer gradients
                dL_dh = [0.0 for _ in range(self.hidden_size)]
                for j in range(self.hidden_size):
                    s = sum(dL_dy[i] * self.W2[i][j] for i in range(self.input_size))
                    dL_dh[j] = s * dsigmoid(h[j])
                # Encoder weights update
                for j in range(self.hidden_size):
                    for k in range(self.input_size):
                        self.W1[j][k] -= lr * dL_dh[j] * x[k]
                    self.b1[j] -= lr * dL_dh[j]
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.6f}")

    def reconstruct(self, x):
        _, y = self.forward(x)
        return y

if __name__ == "__main__":
    # Load spatiotemporal matrix
    with open("spatiotemporal_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    input_size = len(data[0])
    hidden_size = 32  # You can adjust this
    autoencoder = Autoencoder(input_size, hidden_size)
    autoencoder.train(data, epochs=20, lr=0.01)
    # Save trained model weights
    with open("autoencoder_weights.pkl", "wb") as f:
        pickle.dump({
            'W1': autoencoder.W1,
            'b1': autoencoder.b1,
            'W2': autoencoder.W2,
            'b2': autoencoder.b2
        }, f)
    print("Autoencoder training complete and weights saved.")
