# coding:utf-8
import random
from os import listdir

import numpy as np

from PIL import Image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff_sigmoid(x):
    val = sigmoid(x)
    return val * (1 - val)


def tanh(x):
    return np.tanh(x)


def diff_tanh(x):
    val = tanh(x)
    return 1 - val * val


def linear(x):
    return x


def diff_linear(x):
    return np.ones_like(x)


def ReLu(x):
    return (np.abs(x) + x) / 2.0


def diff_ReLu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def softmax(x):
    val = np.exp(x)
    # val -= np.max(val, axis=1, keepdims=True)
    return val / np.sum(val, axis=1, keepdims=True)


def test_rate(X, Y, bp):
    predict = bp.predict(X)
    count = 0
    for i in range(len(X)):
        predict_type = np.argmax(predict[i])
        real_type = np.argmax(Y[i])
        if predict_type == real_type:
            count += 1
    print("总个数：%d,正确个数：%d" % (len(X), count))
    print("正确率：%f" % (count / len(X)))
    return count / len(X)


def cross_entropy(P, Y):
    a = np.array(P).reshape(-1)
    y = np.array(Y).reshape(-1)
    # print(a[:6])
    # print(y[:5])
    return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))) / len(a) * P.shape[1]


def dropout(x, p):
    randomMask = []
    for i in range(x.shape[1]):
        r = random.random()
        if r < p:
            randomMask.append(0)
        else:
            randomMask.append(1 / (1 - p))
    return np.array(randomMask)


class BP:
    def __init__(self, f_hidden='sigmoid', f_output='sigmoid',
                 epsilon=1e-3, maxstep=1000, alpha=0.1, momentum=0.0, batch_size=1, p=0.5):
        self.batch_size = batch_size  # mini_batch
        self.n_input = None  # 输入层神经元数目
        self.n_hidden = []  # 隐藏层神经元数目
        self.n_output = None  # 输出层神经元数目
        self.f_hidden = f_hidden  # 隐藏层输入函数
        self.f_output = f_output  # 输出层输出函数
        self.epsilon = epsilon  # 误差阈值
        self.maxstep = maxstep  # 最大迭代次数
        self.alpha = alpha  # 学习率
        self.momentum = momentum  # 动量因子
        self.p = p  # dropout rate

        self.weight = []  # 每层之间的权重矩阵
        self.bias = []  # 每层之间的偏执向量
        self.N = None  # 输入层神经元的维数=样本数
        self.test_data = None
        self.test_result = None
        self.correct_rate = []
        self.randomMask = []

    ##初始化
    def init_param(self, X_data, y_data, num_list):
        n_layer = len(num_list)
        if n_layer < 3:
            raise ValueError('the length of num_list cannot be less than 3 ')
        # 初始化
        if len(X_data.shape) == 1:  # 行向量->列向量
            X_data = np.transpose([X_data])
        self.N = X_data.shape[0]
        if len(y_data.shape) == 1:  # 行向量->列向量
            y_data = np.transpose([y_data])
        n_input = num_list[0]
        n_output = num_list[n_layer - 1]
        self.n_input = X_data.shape[1]
        self.n_output = y_data.shape[1]
        if n_input != self.n_input or n_output != self.n_output:
            raise ValueError('the dimension of input or output is not same as num_list ')
        self.n_hidden = num_list[1:n_layer - 1]
        # if self.n_hidden is None:
        #     self.n_hidden = int(math.ceil(math.sqrt(self.n_input + self.n_output)) + 2)
        self.weight.append(1e-2 * np.random.randn(self.n_input, self.n_hidden[0]))
        self.bias.append(1e-2 * np.random.randn(self.n_hidden[0]))
        for i in range(len(self.n_hidden) - 1):
            self.weight.append(1e-2 * np.random.randn(self.n_hidden[i], self.n_hidden[i + 1]))
            self.bias.append(1e-2 * np.random.randn(self.n_hidden[i + 1]))
        self.weight.append(1e-2 * np.random.randn(self.n_hidden[-1], self.n_output))
        self.bias.append(1e-2 * np.random.randn(self.n_output))
        return X_data, y_data

    def init_testdata(self, test_data, test_result):
        self.test_data = test_data
        self.test_result = test_result

    def inspirit(self, name):
        # 获取相应的激励函数
        if name == 'sigmoid':
            return sigmoid
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tanh
        elif name == 'softmax':
            return softmax
        elif name == 'ReLu':
            return ReLu
        else:
            raise ValueError('the function is not supported now')

    def diff_inspirit(self, name):
        # 获取相应的激励函数的导数
        if name == 'sigmoid':
            return diff_sigmoid
        elif name == 'linear':
            return diff_linear
        elif name == 'tanh':
            return diff_tanh
        elif name == 'ReLu':
            return diff_ReLu
        else:
            raise ValueError('the function is not supported now')

    def forward(self, X_data, model=False):
        if model:
            r1 = dropout(X_data, self.p)
            self.randomMask.append(r1)
            x_out = [X_data * r1]
            x_in = []
            x_hidden_in = x_out[0] @ self.weight[0] + self.bias[0]  # n*h
            x_hidden_out = self.inspirit(self.f_hidden)(x_hidden_in)  # n*h
            r2 = dropout(x_hidden_out, self.p)
            x_hidden_out *= r2
            self.randomMask.append(r2)
            x_in.append(x_hidden_in)
            x_out.append(x_hidden_out)
            for i in range(len(self.n_hidden) - 1):
                x_hidden_in = x_hidden_out @ self.weight[i + 1] + self.bias[i + 1]  # n*h
                x_hidden_out = self.inspirit(self.f_hidden)(x_hidden_in)  # n*h
                r3 = dropout(x_hidden_out, self.p)
                x_hidden_out *= r3
                self.randomMask.append(r3)
                x_in.append(x_hidden_in)
                x_out.append(x_hidden_out)
            x_output_in = x_hidden_out @ self.weight[-1] + self.bias[-1]  # n*o
            x_output_out = self.inspirit(self.f_output)(x_output_in)  # n*o
            x_in.append(x_output_in)
            x_out.append(x_output_out)
            return x_in, x_out
        # 前向传播
        x_out = [X_data]
        x_in = []
        x_hidden_in = X_data @ self.weight[0] + self.bias[0]  # n*h
        x_hidden_out = self.inspirit(self.f_hidden)(x_hidden_in)  # n*h
        x_in.append(x_hidden_in)
        x_out.append(x_hidden_out)
        for i in range(len(self.n_hidden) - 1):
            x_hidden_in = x_hidden_out @ self.weight[i + 1] + self.bias[i + 1]  # n*h
            x_hidden_out = self.inspirit(self.f_hidden)(x_hidden_in)  # n*h
            x_in.append(x_hidden_in)
            x_out.append(x_hidden_out)
        x_output_in = x_hidden_out @ self.weight[-1] + self.bias[-1]  # n*o
        x_output_out = self.inspirit(self.f_output)(x_output_in)  # n*o
        x_in.append(x_output_in)
        x_out.append(x_output_out)
        return x_in, x_out

    def fit(self, X_data, y_data, num_list):
        # 训练主函数
        X_data, y_data = self.init_param(X_data, y_data, num_list)
        step = 0
        batch_size = self.batch_size
        num_batches = int(self.N / batch_size) + 1
        model = False
        while step < self.maxstep:  # print("enpoch now is %s" % enpoch)
            step += 1
            if step == 5:
                model = True
            x_in, x_out = self.forward(X_data)
            Loss = cross_entropy(x_out[-1], y_data)
            x_in2, x_out2 = self.forward(self.test_data)
            Loss2 = cross_entropy(x_out2[-1], self.test_result)
            print('step:%d, TrainLoss:%s, TestLoss:%s' % (step, Loss, Loss2))
            rate = test_rate(self.test_data, self.test_result, self)
            self.correct_rate.append(rate)
            # print(x_out)
            if Loss < self.epsilon:
                return
            shuffled_order = np.arange(X_data.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)
            # print(shuffled_order)
            # Batch update
            self.delta_weight = np.zeros_like(self.weight)
            self.delta_bias = np.zeros_like(self.bias)
            for batch in range(num_batches):  # 每次迭代要使用的数据量
                test = np.arange(batch_size * batch, batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标
                batch_X = np.array(X_data[shuffled_order[batch_idx]])
                batch_Y = np.array(y_data[shuffled_order[batch_idx]])
                self.backpropagation(batch_X, batch_Y, model)
                # self.update()
            self.alpha *= 0.95 + 0.05 * random.random()
        return

    def backpropagation(self, X_data, y_data, model=False):
        N = X_data.shape[0]
        # 向前传播
        x_in, x_out = self.forward(X_data, model)
        # print(np.argmax(x_out))
        # 误差反向传播，依据权值逐层计算当层误差
        crossentropy_diff = -np.divide(y_data, x_out[-1])  # 100*12
        softmax_diff = np.zeros((N, self.n_output))
        delta_weight = np.zeros_like(self.weight)
        delta_bias = np.zeros_like(self.bias)
        for t in range(N):
            temp = np.zeros((self.n_output, self.n_output))
            for i in range(self.n_output):
                for j in range(self.n_output):
                    if i == j:
                        temp[i][j] = x_out[-1][t][i] * (1 - x_out[-1][t][i])
                    else:
                        temp[i][j] = x_out[-1][t][i] * (- x_out[-1][t][j])
            softmax_diff[t] = crossentropy_diff[t] @ temp
            delta_weight[-1] += np.array([x_out[-2][t]]).T @ np.array([softmax_diff[t]])
            delta_bias[-1] += softmax_diff[t]
        err_hidden = softmax_diff @ self.weight[-1].T  # n*h， 隐藏层，每个神经元上的误差
        # 隐藏层到输出层权值及阈值更新
        self.weight[-1] -= self.alpha * delta_weight[-1] / N + self.momentum * delta_weight[-1]
        self.bias[-1] -= self.alpha * delta_bias[-1][0] / N + self.momentum * delta_bias[-1]
        # 隐藏层到隐藏层以及输入层到隐藏层权值及阈值更新
        for i in range(len(self.n_hidden)):
            if model:
                delta_out = err_hidden * self.diff_inspirit(self.f_hidden)(x_in[-2 - i]) * self.randomMask[-1 - i]
                err_hidden = delta_out @ self.weight[-2 - i].T  # 下一个隐藏层，每个神经元上的误差
                delta_bias[-2 - i] = np.sum(self.alpha * delta_out, axis=0) / N + self.momentum * delta_bias[-2 - i]
                self.bias[-2 - i] -= delta_bias[-2 - i]
                delta_weight[-2 - i] = self.alpha * x_out[-3 - i].T @ delta_out / N + self.momentum * delta_weight[
                    -2 - i]
                self.weight[-2 - i] -= delta_weight[-2 - i]
            else:
                delta_out = err_hidden * self.diff_inspirit(self.f_hidden)(x_in[-2 - i])  # n*o
                err_hidden = delta_out @ self.weight[-2 - i].T  # 下一个隐藏层，每个神经元上的误差
                delta_bias[-2 - i] = np.sum(self.alpha * delta_out, axis=0) / N + self.momentum * delta_bias[-2 - i]
                self.bias[-2 - i] -= delta_bias[-2 - i]
                delta_weight[-2 - i] = self.alpha * x_out[-3 - i].T @ delta_out / N + self.momentum * delta_weight[
                    -2 - i]
                self.weight[-2 - i] -= delta_weight[-2 - i]

    def update(self):
        self.weight -= self.delta_weight
        self.bias -= self.delta_bias

    def predict(self, X):
        # 预测
        res = self.forward(X)
        return res[1][-1]


def change_image_to_matrix(image_file):
    ret = np.zeros(28 * 28)
    image = Image.open(image_file).convert('RGB')
    for i in range(28):
        for j in range(28):
            ret[28 * i + j] = int(image.getpixel((i, j))[0] / 255)
    return ret


def load_certain_type_images(filename, type):
    X = []
    Y = []
    y = np.zeros(12)
    y[type] = 1
    filesname = listdir(filename)
    for i in range(len(filesname)):
        X.append(change_image_to_matrix(filename + "/" + filesname[i]))
        Y.append(y)
    return X, Y


def load_all_images(filename):
    X = []
    Y = []
    filesname = listdir(filename)
    for i in range(len(filesname)):
        x, y = load_certain_type_images(filename + "/" + filesname[i], i)
        X += x
        Y += y
    print("Data loaded successfully")
    return np.array(X), np.array(Y)


def store(input, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(input, fw)
    fw.close()


def grab(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)


def split_and_test():
    X_data = grab("X_data.save")
    Y_data = grab("Y_data.save")
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(1, len(X_data) + 1):
        if i % 620 == 0:
            type = int(i / 620)
            shuffled_order = np.arange((type - 1) * 620, type * 620)  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)
            test = np.arange(0, int(0.8 * 620))
            test2 = np.arange(int(0.8 * 620), 620)
            # train_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标
            train_X += (X_data[shuffled_order[test]]).tolist()
            train_Y += (Y_data[shuffled_order[test]]).tolist()
            test_X += (X_data[shuffled_order[test2]]).tolist()
            test_Y += (Y_data[shuffled_order[test2]]).tolist()
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    # 28-38
    results = []
    for i in range(160, 161, 10):
        bp = BP(f_hidden='ReLu', f_output='softmax', maxstep=150, alpha=0.01, momentum=0,
                batch_size=4, p=0.1)  # 注意学习率若过大，将导致不能收敛
        # test_X, test_Y = load_all_images("./test")
        bp.init_testdata(test_X, test_Y)
        bp.fit(train_X, train_Y, [28 * 28, 180, 180, 12])
        results.append((np.argmax(bp.correct_rate), np.max(bp.correct_rate)))
        print(results)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    split_and_test()
    # X_data, Y_data = load_all_images("./train")
    # store(X_data,"X_data.save")
    # store(Y_data,"Y_data.save"x   )

    # test(test_X, test_Y, bp)
    # x = change_image_to_matrix("./train/1/1.bmp")
    # print(x)
    # x,y = loadImage("./train/1",0)
    # print(x)
    # print(y)
