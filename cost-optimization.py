import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]

# W = tf.Variable(tf.random.normal([1]), name='weight')

# hypothesis = X * W
# cost = tf.reduce_sum(tf.square(hypothesis - Y))


# learning_rate = 0.1
# for step in range(21):
#     gradient = tf.reduce_mean((W * X - Y) * X)
#     descent = W - learning_rate * gradient      
#     update = W.assign(descent)
#     print(step, update.numpy(), W.numpy(), gradient.numpy())

W = tf.Variable(5., type=float)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


for step in range(101):
     with tf.GradientTape() as tape:
            hypothesis = X * W
            cost = tf.reduce_mean(tf.square(hypothesis - Y))            
            gradients = tape.gradient(cost, [W])
            optimizer.apply_gradients(zip(gradients, [W]))
            print(step, cost.numpy(), W.numpy())
