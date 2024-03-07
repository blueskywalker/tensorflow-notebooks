import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def main():
    # X = tf.Variable([1., 2., 3.])
    # Y = tf.Variable([1., 2., 3.])
    #
    # W = tf.Variable(tf.random.normal([1]), name='weight')
    #
    # hypothesis = tf.multiply(X, W)
    # cost = tf.reduce_sum(tf.square(hypothesis - Y))
    #
    # learning_rate = 0.1
    # for step in range(21):
    #     gradient = tf.reduce_mean((W * X - Y) * X)
    #     descent = W - learning_rate * gradient
    #     update = W.assign(descent)
    #     print(step, update.numpy(), W.numpy(), gradient.numpy())

    X = np.array([1., 2., 3.])
    Y = np.array([1., 2., 3.])

    W = tf.Variable(tf.random.normal([1]), name='weight')

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    print(X.shape, W.shape)
    for step in range(101):
        with tf.GradientTape() as tape:
            hypothesis = X * W
            cost = tf.reduce_mean(tf.square(hypothesis - Y))
            gradients = tape.gradient(cost, [W])
            optimizer.apply_gradients(zip(gradients, [W]))
            print(step, cost.numpy(), W.numpy())


if __name__ == '__main__':
    main()

