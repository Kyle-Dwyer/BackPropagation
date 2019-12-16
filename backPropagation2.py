# coding:utf-8
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


class BP:
    def __init__(self, f_hidden='sigmoid', f_output='sigmoid',
                 epsilon=1e-3, maxstep=1000, alpha=0.1, momentum=0.0):
        self.n_input = 0  # 输入层神经元数目
        self.n_hidden = []  # 隐藏层神经元数目
        self.n_output = 0  # 输出层神经元数目
        self.f_hidden = f_hidden  # 隐藏层输入函数
        self.f_output = f_output  # 输出层输出函数
        self.epsilon = epsilon  # 误差阈值
        self.maxstep = maxstep  # 最大迭代次数
        self.alpha = alpha  # 学习率
        self.momentum = momentum  # 动量因子

        self.weight = []  # 每层之间的权重矩阵
        self.bias = []  # 每层之间的偏执向量
        self.N = 0  # 输入层神经元的维数=样本数

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
        self.weight.append(np.random.uniform(-1, 1, (self.n_input, self.n_hidden[0])))
        self.bias.append(np.random.uniform(-1, 1, self.n_hidden[0]))
        for i in range(len(self.n_hidden) - 1):
            self.weight.append(np.random.uniform(-1, 1, (self.n_hidden[i], self.n_hidden[i + 1])))
            self.bias.append(np.random.uniform(-1, 1, self.n_hidden[i + 1]))
        self.weight.append(np.random.uniform(-1, 1, (self.n_hidden[-1], self.n_output)))
        self.bias.append(np.random.uniform(-1, 1, self.n_output))
        return X_data, y_data

    def inspirit(self, name):
        # 获取相应的激励函数
        if name == 'sigmoid':
            return sigmoid
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tanh
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
        else:
            raise ValueError('the function is not supported now')

    def forward(self, X_data):
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
        # 初始化动量项
        delta_weight = np.zeros_like(self.weight)
        delta_bias = np.zeros_like(self.bias)
        while step < self.maxstep:
            step += 1
            # 向前传播
            x_in, x_out = self.forward(X_data)
            error_avg = np.sum(abs(x_out[-1] - y_data))
            print(error_avg)
            if error_avg < self.epsilon:
                break
            # 误差反向传播，依据权值逐层计算当层误差
            err_output = y_data - x_out[-1]  # n*o， 输出层上，每个神经元上的误差
            delta_out = -err_output * self.diff_inspirit(self.f_output)(x_in[-1])  # n*o
            err_hidden = delta_out @ self.weight[-1].T  # n*h， 隐藏层，每个神经元上的误差
            # 隐藏层到输出层权值及阈值更新
            delta_bias[-1] = np.sum(self.alpha * delta_out + self.momentum * delta_bias[-1], axis=0) / self.N
            self.bias[-1] -= delta_bias[-1]
            delta_weight[-1] = self.alpha * x_out[-2].T @ delta_out + self.momentum * delta_weight[-1]
            self.weight[-1] -= delta_weight[-1]
            # 隐藏层到隐藏层以及输入层到隐藏层权值及阈值更新
            for i in range(len(self.n_hidden)):
                delta_out = err_hidden * self.diff_inspirit(self.f_hidden)(x_in[-2 - i])  # n*o
                err_hidden = delta_out @ self.weight[-2 - i].T  # 下一个隐藏层，每个神经元上的误差
                delta_bias[-2 - i] = np.sum(self.alpha * delta_out + self.momentum * delta_bias[-2-i], axis=0) / self.N
                self.bias[-2 - i] -= delta_bias[-2 - i]
                delta_weight[-2 - i] = self.alpha * x_out[-3 - i].T @ delta_out + self.momentum * delta_weight[-2 - i]
                self.weight[-2 - i] -= delta_weight[-2 - i]
        print(step)
        return

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


def test(X, Y, bp):
    predict = bp.predict(X)
    count = 0
    for i in range(len(X)):
        predict_type = np.argmax(predict[i])
        real_type = np.argmax(Y[i])
        if predict_type == real_type:
            count += 1
    print("总个数：%d,正确个数：%d" % (len(X), count))
    print("正确率：%f" % (count / len(X)))


def store(input, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(input, fw)
    fw.close()


def grab(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X_data, Y_data = load_all_images("./train")
    bp = BP(f_hidden='sigmoid', f_output='sigmoid', maxstep=5000, alpha=0.001, momentum=0.5)  # 注意学习率若过大，将导致不能收敛
    bp.fit(X_data, Y_data, [28*28, 150, 150, 12])
    test_X, test_Y = load_all_images("./test")
    test(test_X, test_Y, bp)
    # x = change_image_to_matrix("./train/1/1.bmp")
    # print(x)
    # x,y = loadImage("./train/1",0)
    # print(x)
    # print(y)

# 该算法给出的是可控神经元个数的单隐藏层BP算法
# 每次运行可能有浮动，这是因为最开始生成的W和B是随机的，结束判定时的情况可能不同，而且有时候是局部最优解
# 回归用linear，分类用sigmoid

# BP算法改进的主要目标是加快训练速度，避免陷入局部极小值等，常见的改进方法有带动量因子算法、自适应学习速率、变化的学习速率以及作用
# 函数后缩法等。 动量因子法的基本思想是在反向传播的基础上，在每一个权值的变化上加上一项正比于前次权值变化的值，并根据反向传播法来
# 产生新的权值变化。而自适应学习 速率的方法则是针对一些特定的问题的。改变学习速率的方法的原则是，若连续几次迭代中，若目标函数对某
# 个权倒数的符号相同，则这个权的学习速率增加， 反之若符号相反则减小它的学习速率。而作用函数后缩法则是将作用函数进行平移，即加上一个常数。
