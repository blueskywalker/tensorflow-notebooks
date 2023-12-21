import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]

W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    hypothesis = X * tf.constant(feed_W) 
    curr_cost = tf.reduce_mean(tf.square(hypothesis - Y))
    W_val.append(feed_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
