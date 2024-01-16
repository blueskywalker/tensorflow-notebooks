import tensorflow as tf
import keras
import numpy as np

mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

one_batch = 200

def get_train_batch_data(size: int):
    def batch(n: int):
        start = n * size
        end = start + size
        return x_train[start:end], y_train[start:end]

    return batch

def train():
    next_batch = get_train_batch_data(one_batch)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    epochs = 20    
    for epoch in range(epochs):
        avg_loss = 0.
        batch_total = x_train.shape[0] // one_batch
        for n in range(batch_total):
            x, y = next_batch(n)

            with tf.GradientTape() as tape:
                # layer 1                
                l1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
                l1 = tf.nn.relu(l1)
                l1 = tf.nn.max_pool2d(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='SAME')
                # 14 x 14 (28/2)
                # layer 2
                l2 = tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME')
                l2 = tf.nn.relu(l2)
                l2 = tf.nn.max_pool2d(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='SAME')
                # 7 x 7 (14/2)
                # flatten                
                l2 = tf.reshape(l2, [-1, flatten_size])

                # final FC 7 * 7 * 64 inputs -> 10 outputs
                h = tf.matmul(l2, W3) + b
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
                grads = tape.gradient(cost, [W3, b, W2, W1])
                optimizer.apply_gradients(grads_and_vars=zip(grads, [W3, b, W2, W1]))
                avg_loss += cost / batch_total
        print(f'epoch: {epoch}, loss: {avg_loss}')

flatten_size = 7 * 7 * 64
W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01), name="W1")
W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01), name="W2")
initializer = tf.keras.initializers.GlorotNormal()
W3 = tf.Variable(initializer([flatten_size, 10]), name="W3")
b = tf.Variable(tf.random.normal([10]), name="b")


# evaluate
def evaluate():
    # layer 1
    l1 = tf.nn.conv2d(x_test, W1, strides=[1, 1, 1, 1], padding='SAME')
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.max_pool2d(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='SAME')
    # 14 x 14 (28/2)
    # layer 2
    l2 = tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.max_pool2d(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='SAME')
    # 7 x 7 (14/2)
    # flatten                
    l2 = tf.reshape(l2, [-1, flatten_size])

    # final FC 7 * 7 * 64 inputs -> 10 outputs
    h = tf.matmul(l2, W3) + b
    pred = tf.argmax(h, 1)
    correct = tf.equal(pred, tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print(f'accuracy: {accuracy}')

train()
evaluate()