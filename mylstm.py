from tqdm import tqdm
import numpy as np
import re

data = """
우리 나라 말이 중국과 달라 한자와는 서로 말이 통하지 아니하여서 이런 까닭으로 어리석은 백성이 말하고자 할 것이 있어도 마침내 제 뜻을 펴지 못하는 사람이 많다. 내가 이것을 가엾게 여겨 새로 스물 여덟 글자를 만드니 모든 사람들로 하여금 쉽게 익혀서 날마다 쓰는 데 편하게 하고자 할 따름이니라."""


# 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x, derivative=False):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x ** 2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def data_preprocessing(src):
    src = re.sub('[^가-힣]', ' ', src)
    t = data.split()
    vocab = list(set(t))
    size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    return t, size, word_to_ix, ix_to_word


tokens, vocab_size, Word_to_ix, ix_to_Word = data_preprocessing(data)


class LSTM:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        # Forget Gate
        self.Wf = np.random.randn(hidden_size, input_size)*0.1
        self.bf = np.zeros((hidden_size, 1))

        # Input Gate
        self.Wi = np.random.randn(hidden_size, input_size)*0.1
        self.bi = np.zeros((hidden_size, 1))

        # Candidate Gate
        self.Wc = np.random.randn(hidden_size, input_size)*0.1
        self.bc = np.zeros((hidden_size, 1))

        # Output Gate
        self.Wo = np.random.randn(hidden_size, input_size)*0.1
        self.bo = np.zeros((hidden_size, 1))

        # Final Gate
        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    # 네트워크 메모리 리셋
    def reset(self):
        self.X = {}

        self.HS = {-1: np.zeros((self.hidden_size, 1))}
        self.CS = {-1: np.zeros((self.hidden_size, 1))}

        self.C = {}
        self.O = {}
        self.F = {}
        self.I = {}
        self.outputs = {}

    # Forward 순전파
    def forward(self, inputs):
        # self.reset()
        x = {}
        outputs = []
        for t in range(len(inputs)):
            x[t] = np.zeros((vocab_size , 1))
            x[t][inputs[t]] = 1  # 각각의 Word에 대한 one hot coding
            self.X[t] = np.concatenate((self.HS[t - 1], x[t]))

            self.F[t] = sigmoid(np.dot(self.Wf, self.X[t]) + self.bf)
            self.I[t] = sigmoid(np.dot(self.Wi, self.X[t]) + self.bi)
            self.C[t] = tanh(np.dot(self.Wc, self.X[t]) + self.bc)
            self.O[t] = sigmoid(np.dot(self.Wo, self.X[t]) + self.bo)

            self.CS[t] = self.F[t] * self.CS[t - 1] + self.I[t] * self.C[t]
            self.HS[t] = self.O[t] * tanh(self.CS[t])

            outputs += [np.dot(self.Wy, self.HS[t]) + self.by]

        return outputs

    # 역전파
    def backward(self, errors, inputs):
        dLdWf, dLdbf = 0, 0
        dLdWi, dLdbi = 0, 0
        dLdWc, dLdbc = 0, 0
        dLdWo, dLdbo = 0, 0
        dLdWy, dLdby = 0, 0

        dh_next, dc_next = np.zeros_like(self.HS[0]), np.zeros_like(self.CS[0])
        for t in reversed(range(len(inputs))):
            error = errors[t]

            # Final Gate Weights and Biases Errors
            dLdWy += np.dot(error, self.HS[t].T)         #𝜕𝐿/𝜕𝑊𝑦
            dLdby += error                               #𝜕𝐿/𝜕b𝑦 = (𝜕𝐿/𝜕z_t)(𝜕z_t/𝜕b𝑦) = error x 1 (Zt = WyHSt + by)

            # Hidden State Error
            dLdHS = np.dot(self.Wy.T, error) + dh_next    #𝜕𝐿/𝜕𝐻𝑆

            # Output Gate Weights and Biases Errors
            dLdo = tanh(self.CS[t]) * dLdHS * sigmoid_derivative(self.O[t])
            dLdWo += np.dot(dLdo, inputs[t].T)
            dLdbo += dLdo

            # Cell State Error
            dLdCS = tanh_derivative(tanh(self.CS[t])) * self.O[t] * dLdHS + dc_next

            # Forget Gate Weights and Biases Errors
            dLdf = dLdCS * self.CS[t - 1] * sigmoid_derivative(self.F[t])
            dLdWf += np.dot(dLdf, inputs[t].T)
            dLdbf += dLdf

            # Input Gate Weights and Biases Errors
            dLdi = dLdCS * self.C[t] * sigmoid_derivative(self.I[t])
            dLdWi += np.dot(dLdi, inputs[t].T)
            dLdbi += dLdi

            # Candidate Gate Weights and Biases Errors
            dLdc = dLdCS * self.I[t] * tanh_derivative(self.C[t])
            dLdWc += np.dot(dLdc, inputs[t].T)
            dLdbc += dLdc

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = np.dot(self.Wf.T, dLdf) + np.dot(self.Wi.T, dLdi) + np.dot(self.Wc.T, dLdc) + np.dot(self.Wo.T, dLdo)

            # Error of Hidden State and Cell State at Next Time Step
            dh_next = d_z[:self.hidden_size, :]
            dc_next = self.F[t] * dLdCS

        for d_ in (dLdWf, dLdbf, dLdWi, dLdbi, dLdWc, dLdbc, dLdWo, dLdbo, dLdWy, dLdby):
            np.clip(d_, -1, 1, out=d_)


        self.Wf += dLdWf * self.learning_rate * (-1)
        self.bf += dLdbf * self.learning_rate * (-1)

        self.Wi += dLdWi * self.learning_rate * (-1)
        self.bi += dLdbi * self.learning_rate * (-1)

        self.Wc += dLdWc * self.learning_rate * (-1)
        self.bc += dLdbc * self.learning_rate * (-1)

        self.Wo += dLdWo * self.learning_rate * (-1)
        self.bo += dLdbo * self.learning_rate * (-1)

        self.Wy += dLdWy * self.learning_rate * (-1)
        self.by += dLdby * self.learning_rate * (-1)

    # Train
    def train(self, inputs, labels):
        for _ in tqdm(range(self.num_epochs)):
            self.reset()
            input_idx = [Word_to_ix[input] for input in inputs]
            predictions = self.forward(input_idx)

            errors = []
            for t in range(len(predictions)):
                errors += [softmax(predictions[t])]
                errors[-1][Word_to_ix[labels[t]]] -= 1

            self.backward(errors, self.X)

    def test(self, inputs, labels):
        accuracy = 0
        probabilities = self.forward([Word_to_ix[input] for input in inputs])

        gt = ''
        output = '우리 '
        for q in range(len(labels)):
            prediction = ix_to_Word[np.argmax(softmax(probabilities[q].reshape(-1)))]
            gt += inputs[q] + ' '
            output += prediction + ' '

            if prediction == labels[q]:
                accuracy += 1

        print('실제값: ', gt)
        print('예측값: ', output)


def main():
    hidden_size = 25
    lstm = LSTM(input_size=vocab_size+hidden_size, hidden_size=hidden_size, output_size=vocab_size, num_epochs=1000, learning_rate=0.05)
    train_X, train_y = tokens[:-1], tokens[1:]
    lstm.train(train_X, train_y)
    lstm.test(train_X, train_y)


if __name__ == '__main__':
    main()
