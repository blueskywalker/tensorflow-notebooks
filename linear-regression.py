import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.keras.optimizers.SGD(0.5)

# Run the training for 201 steps
for step in range(201):
    with tf.GradientTape() as tape:
        y = W * x_data + b
        loss = tf.reduce_mean(tf.square(y - y_data))
        gradients = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))
        if step % 20 == 0:
            print(step, W.numpy(), b.numpy())
