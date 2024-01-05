# Import NumPy
import numpy as np


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Define the input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Define the output data
y = np.array([[0],
              [1],
              [1],
              [0]])

# Set the random seed
np.random.seed(42)

# Initialize the weights randomly
weights1 = 2 * np.random.random((3, 3)) - 1  # weights for the hidden layer
weights2 = 2 * np.random.random((3, 1)) - 1  # weights for the output layer

# Set the learning rate
alpha = 0.1

# Train the neural network
for epoch in range(10000):

    # Forward propagation
    layer1 = sigmoid(np.dot(X, weights1))  # output of the hidden layer
    layer2 = sigmoid(np.dot(layer1, weights2))  # output of the output layer

    # Backpropagation
    layer2_error = y - layer2  # error of the output layer
    layer2_delta = layer2_error * sigmoid_derivative(layer2)  # delta of the output layer
    layer1_error = np.dot(layer2_delta, weights2.T)  # error of the hidden layer
    layer1_delta = layer1_error * sigmoid_derivative(layer1)  # delta of the hidden layer

    # Update the weights
    weights2 += alpha * np.dot(layer1.T, layer2_delta)  # update the weights of the output layer
    weights1 += alpha * np.dot(X.T, layer1_delta)  # update the weights of the hidden layer

    # Print the error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Error: {np.mean(np.abs(layer2_error))}")

# Print the final output
print(f"Output: {layer2}")
