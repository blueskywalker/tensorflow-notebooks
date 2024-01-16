import numpy as np
import re

data = """
우리 나라 말이 중국과 달라 한자와는 서로 말이 통하지 아니하여서 이런 까닭으로 어리석은 백성이 말하고자 할 것이 있어도 마침내 제 뜻을 펴지 못하는 사람이 많다. 내가 이것을 가엾게 여겨 새로 스물 여덟 글자를 만드니 모든 사람들로 하여금 쉽게 익혀서 날마다 쓰는 데 편하게 하고자 할 따름이니라."""

class MyRNN(object):
    def __init__(self, source, hidden_dim, seq_length, learning_rate=0.01):
        np.random.seed(223345)
        self.word_to_idx = None
        self.idx_to_word = None
        self.vocab = None
        self.tokens = None
        self.data_preprocessing(source)
        self.hidden_size = hidden_dim
        self.input_size = self.output_size = len(self.vocab)        
        self.seq_length = seq_length
        self.W_xh = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.W_hy = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b_h = np.zeros((1, self.hidden_size))
        self.b_y = np.zeros((1, self.output_size))
        self.learning_rate = learning_rate

    def data_preprocessing(self, source):
        source = re.sub(r'[^가-힣]', '', source)
        self.tokens = data.split()
        self.vocab = list(set(self.tokens))        
        self.word_to_idx = {w:i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i:w for i, w in enumerate(self.vocab)}


    def one_hot_encoding(self, idx):
        one_hot_vector = np.zeros((1, self.input_size))
        one_hot_vector[0][idx] = 1
        return one_hot_vector
    
    def forward(self, inputs, target, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        for t in range(self.seq_length):
            xs[t] = self.one_hot_encoding(inputs[t])            
            hs[t] = np.tanh(np.dot(xs[t], self.W_xh) + np.dot(hs[t-1], self.W_hh) + self.b_h)
            ys[t] = np.dot(hs[t], self.W_hy) + self.b_y
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][0][target[t]])
        return loss, ps, hs, xs

    def predict(self, word, length):
        h_prev = np.zeros((1, self.hidden_size))
        x = self.one_hot_encoding(self.word_to_idx[word])
        idxs = []
        for t in range(length):
            h = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
            y = np.dot(h, self.W_hy) + self.b_y
            p = np.exp(y) / np.sum(np.exp(y))
            idx = np.random.choice(range(self.input_size), p=p.ravel())
            idxs.append(idx)
            x = self.one_hot_encoding(idx)
            h_prev = h
        return ' '.join([self.idx_to_word[i] for i in idxs])
    

    ## backpropagation
    def backward(self, ps, hs, xs, target):
        dW_xh, dW_hh, dW_hy = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy)
        db_h, db_y = np.zeros_like(self.b_h), np.zeros_like(self.b_y)
        dh_next = np.zeros_like(hs[0])
        for t in reversed(range(self.seq_length)):
            dy = np.copy(ps[t])
            dy[0][target[t]] -= 1
            dW_hy += np.dot(hs[t].T, dy)
            db_y += dy
            dh = np.dot(dy, self.W_hy.T) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh
            db_h += dh_raw
            dW_xh += np.dot(xs[t].T, dh_raw)
            dW_hh += np.dot(hs[t-1].T, dh_raw)
            dh_next = np.dot(dh_raw, self.W_hh.T)
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)
        return dW_xh, dW_hh, dW_hy, db_h, db_y
        
    def train(self, epochs):
        for epoch in range(epochs):
            h_prev = np.zeros((1, self.hidden_size))
            loss = 0.
            for i in range(len(self.tokens) // self.seq_length):
                inputs = [self.word_to_idx[w] for w in self.tokens[i*self.seq_length:(i+1)*self.seq_length]]
                target = [self.word_to_idx[w] for w in self.tokens[i*self.seq_length+1:(i+1)*self.seq_length+1]]
                loss, ps, hs, xs = self.forward(inputs, target, h_prev)
                dW_xh, dW_hh, dW_hy, db_h, db_y = self.backward(ps, hs, xs, target)
                for param, dparam in zip([self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y], [dW_xh, dW_hh, dW_hy, db_h, db_y]):
                    param += -self.learning_rate * dparam
                h_prev = hs[self.seq_length-1]
            if epoch % 100 == 0:
                print('epoch: %d, loss: %f' % (epoch, loss))
            


def main():
    rnn = MyRNN(data, 100, 25)
    rnn.train(10000)

    response = rnn.predict('나라', 50)
    print(response)

if __name__ == '__main__':
    main()