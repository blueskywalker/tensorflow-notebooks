import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# print(len(x_train), len(x_test))
# print(x_train.shape, x_test.shape)
x_train = tf.reshape(x_train, [x_train.shape[0], -1])

one_batch = 200

def get_train_batch_data(size: int):
    def batch(n: int):
        start = n * size
        end = start + size
        return x_train[start:end], y_train[start:end]

    return batch

def train(w1, bias1, w2, bias2, w3, bias3):
    next_batch = get_train_batch_data(one_batch)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    epochs = 20
    for epoch in range(epochs):
        avg_loss = 0.
        batch_total = x_train.shape[0] // one_batch
        for n in range(batch_total):
            x, y_o = next_batch(n)
            y = tf.one_hot(y_o, depth=10)

            with tf.GradientTape() as tape:
                l1 = tf.nn.relu(tf.matmul(x, w1) + bias1)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + bias2)
                h = tf.matmul(l2, w3) + bias3
                # print(hypothesis.shape, y.shape)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
                grads = tape.gradient(loss, [w3, bias3, w2, bias2, W1, b1])
                optimizer.apply_gradients(grads_and_vars=zip(grads, [w3, bias3, w2, bias2, w1, bias1]))
                avg_loss += loss / batch_total
        print(f'epoch: {epoch}, loss: {avg_loss}')

def initialize(shape):
    xavier = tf.keras.initializers.GlorotNormal()
    return tf.Variable(dtype=tf.float64, initial_value=tf.cast(xavier(shape=shape), dtype=tf.float64))

W1 = initialize([28 * 28, 256])
b1 = tf.Variable(tf.cast(tf.random.normal([256]), dtype=tf.float64), dtype=tf.float64)
W2 = initialize([256, 256])
b2 = tf.Variable(tf.cast(tf.random.normal([256]), dtype=tf.float64), dtype=tf.float64)
W3 = initialize([256, 10])
b3 = tf.Variable(tf.cast(tf.random.normal([10]), dtype=tf.float64), dtype=tf.float64)

train(W1, b1, W2, b2, W3, b3)
print("=" * 50)
test_set = tf.reshape(x_test, [x_test.shape[0], -1])
L1 = tf.nn.relu(tf.matmul(test_set, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
hypothesis = tf.matmul(L2, W3) + b
result_set = tf.one_hot(y_test, depth=10)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(result_set, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: {:.2f} %".format(accuracy.numpy() * 100))
