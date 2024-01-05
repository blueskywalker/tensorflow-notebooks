import pandas as pd
import tensorflow as tf


def softmax_loss(h, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y))


class SoftmaxRegression(tf.Module):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.build = False

    def __call__(self, x, n_cases=1, train=True):
        if not self.build:
            self.w = tf.Variable(tf.random.uniform(shape=[x.shape[-1], n_cases]))
            self.b = tf.Variable(tf.random.uniform(shape=[n_cases]))
            self.build = True

        z = tf.add(tf.matmul(x, self.w), self.b)

        if train:
            return z

        return tf.nn.softmax(z)


def predict_class(y_hat):
    return tf.math.argmax(y_hat)


def accuracy(y_hat, y):
    # y_hat = tf.math.softmax(y_hat)
    return tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(y_hat, 1), tf.math.argmax(y, 1)), tf.float32))


def main():
    data_url = "https://raw.githubusercontent.com/hunkim/DeepLearningZeroToAll/master/data-04-zoo.csv"
    columns = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes",
               "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]
    data = pd.read_csv(data_url, names=columns, comment='#')
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)
    x_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    x_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    y_one_hot = tf.one_hot(indices=y_train, depth=7)
    labels = tf.one_hot(indices=y_test, depth=7)
    softmax_regression = SoftmaxRegression()
    X=tf.convert_to_tensor(x_train, dtype=tf.float32)
    I=tf.convert_to_tensor(x_test, dtype=tf.float32)

    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y_one_hot))
    train_dataset = train_dataset.shuffle(buffer_size=X.shape[0]).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((I, labels))
    test_dataset = test_dataset.shuffle(buffer_size=I.shape[0]).batch(batch_size)

    epochs = 200
    learning_rate = 0.01
    train_losses, test_losses = [], []
    train_corrects, test_corrects = [], []

    for epoch in range(epochs):
        batch_losses_train, batch_correct_train = [], []
        batch_losses_test, batch_correct_test = [], []

        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                y_pred_batch = softmax_regression(x_batch, n_cases=7)
                batch_loss = softmax_loss(y_pred_batch, y_batch)
            batch_correct = accuracy(y_pred_batch, y_batch)
            grads = tape.gradient(batch_loss, softmax_regression.variables)
            for g, v in zip(grads, softmax_regression.variables):
                v.assign_sub(learning_rate * g)
            batch_losses_train.append(batch_loss)
            batch_correct_train.append(batch_correct)

        for x_batch, y_batch in test_dataset:
            y_pred_batch = softmax_regression(x_batch)
            batch_loss = softmax_loss(y_pred_batch, y_batch)
            batch_correct = accuracy(y_pred_batch, y_batch)
            batch_losses_test.append(batch_loss)
            batch_correct_test.append(batch_correct)

        # Keep track of epoch-level model performance
        train_loss, train_correct = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_correct_train)
        test_loss, test_correct = tf.reduce_mean(batch_losses_test), tf.reduce_mean(batch_correct_test)
        train_losses.append(train_loss)
        train_corrects.append(train_correct)
        test_losses.append(test_loss)
        test_corrects.append(test_correct)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Training log loss: {train_loss:.3f}")


if __name__ == '__main__':
    main()
