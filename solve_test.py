__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
尝试求解一维Burgers方程: u_t+u*u_x=(0.01/pi)*u_xx
x:[-1,1]
t:[0,1]
IC:u(t=0,x) = -sin(pi*x)
BC:u(t,x=-1) = u(t,x=1) = 0

objective function: f = u_t+u*u_x-(0.01/pi)*u_xx
'''

LEARNING_RATE_BASE = 0.0001  # 学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


class NN:
    def __init__(self):
        # 方程相关参数
        self.niu = 0.01 / np.pi  # 粘度
        # 求解域
        self.x_lower_bound = -1
        self.x_upper_bound = 1
        self.t_lower_bound = 0
        self.t_upper_bound = 1

        # 数据集相关参数
        self.INPUT_NODE = 2  # 输入层节点数，[x,t]
        self.OUTPUT_NODE = 1  # 输出层节点数，u
        self.BATCH_SIZE = 1000  # 一个训练batch中的训练数据个数

        # 神经网络相关参数
        self.LAYER_NODE = [50, 50, 50, 50, 50, 50, 50, 50]  # 隐藏层节点数。
        self.NUM_LAYER = len(self.LAYER_NODE)  # 隐藏层数
        # 生成隐藏层的参数
        self.weights = []
        self.biases = []
        self.weights.append(tf.Variable(tf.truncated_normal([self.INPUT_NODE, self.LAYER_NODE[0]], stddev=0.1)))
        self.biases.append(tf.Variable(tf.constant(0.1, shape=[self.LAYER_NODE[0]])))
        for i in range(1, self.NUM_LAYER):
            self.weights.append(
                tf.Variable(tf.truncated_normal([self.LAYER_NODE[i - 1], self.LAYER_NODE[i]], stddev=0.1)))
            self.biases.append(tf.Variable(tf.constant(0.1, shape=[self.LAYER_NODE[i]])))
        # 生成输出层的参数
        self.weights_out = tf.Variable(tf.truncated_normal([self.LAYER_NODE[-1], self.OUTPUT_NODE], stddev=0.1))
        self.biases_out = tf.Variable(tf.constant(0.1, shape=[self.OUTPUT_NODE]))

        self.init_op1 = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init_op1)

    # 初始条件
    def initial_condition(self, x):
        return -tf.sin(np.pi * x)

    # 边界条件
    def boundary_condition(self, t):
        return 0

    def inference(self, input_tensor):
        '''辅助函数，给定网络的输入和所有参数，计算神经网络的前向传播结果。'''
        # 计算输出层前向传播结果。
        layer = []
        layer.append(tf.nn.relu(tf.matmul(input_tensor, self.weights[0]) + self.biases[0]))
        for i in range(1, self.NUM_LAYER):
            layer.append(tf.nn.relu(tf.matmul(layer[i - 1], self.weights[i]) + self.biases[i]))

        return tf.matmul(layer[-1], self.weights_out) + self.biases_out

    # u函数值
    def func_u(self, x, t):
        u = self.inference(tf.concat([x, t], 1))
        return u

    # f函数值
    def obj_f(self, x, t):
        u = self.func_u(x, t)
        # tf.gradients: https://www.tensorflow.org/api_docs/python/tf/gradients
        # return: A list of sum(dy/dx) for each x in xs.
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u * u_x - self.niu * u_xx
        return f

    def test_f(self, x, t):
        u = self.func_u(x, t)
        f = u - tf.sin(np.pi * (t * t + x * x))
        return f

    def get_train_data(self):
        # 生成训练点
        x = np.random.uniform(-1, 1, [self.BATCH_SIZE])
        index = np.random.randint(0, self.BATCH_SIZE, [4])
        x[index[0]] = -1
        x[index[1]] = 1
        t = np.random.uniform(0, 1, [self.BATCH_SIZE])
        t[index[2]] = 0
        t[index[3]] = 1
        train_x = x.reshape([self.BATCH_SIZE, 1])
        train_t = t.reshape([self.BATCH_SIZE, 1])
        y = np.sin(np.pi * (train_x * train_x + train_t * train_t))
        # y = self.test_f(train_x, train_t)
        return train_x, train_t, y

    def get_valid_data(self):
        # 验证点
        a = np.loadtxt('axis.txt')
        x = []
        t = []
        for i in range(self.BATCH_SIZE):
            x.append(a[i][0])
            t.append(a[i][1])
        x = np.array(x).reshape([self.BATCH_SIZE, 1])
        t = np.array(x).reshape([self.BATCH_SIZE, 1])
        y = np.sin(np.pi * (x * x + t * t))
        # y = self.test_f(x, t)
        return x, t, y

    def generate_data(self):
        f = open('axis.txt', 'w')
        x = np.random.uniform(-1, 1, [self.BATCH_SIZE])
        index = np.random.randint(0, self.BATCH_SIZE, [4])
        # x[index[0]] = -1
        # x[index[1]] = 1
        t = np.random.uniform(0, 1, [self.BATCH_SIZE])
        # t[index[2]] = 0
        # t[index[3]] = 1
        for i in range(self.BATCH_SIZE):
            f.write(str(x[i]) + ' ' + str(t[i]) + '\n')
        f.close()

    # 训练模型的过程
    def train(self):
        # 输入节点
        x = tf.placeholder(dtype=tf.float32, shape=(self.BATCH_SIZE, 1), name='x-input')
        t = tf.placeholder(dtype=tf.float32, shape=(self.BATCH_SIZE, 1), name='t-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(self.BATCH_SIZE, 1), name='y-output')

        global_step = tf.Variable(0, trainable=False)

        # zero = tf.Variable(tf.zeros([self.BATCH_SIZE, 1]))
        # ones = tf.Variable(tf.ones([self.BATCH_SIZE, 1]))
        '''loss = tf.reduce_mean(tf.square(self.obj_f(x, t)) +
                              10 * (tf.square(self.func_u(ones, t) - self.boundary_condition(t)) +
                                    tf.square(self.func_u(-ones, t) - self.boundary_condition(t)) +
                                    tf.square(self.func_u(x, zero) - self.initial_condition(x))))'''

        #loss = tf.reduce_mean(tf.square(self.test_f(x, t)) + tf.square(y_ - self.func_u(x, t)))
        #loss = tf.reduce_mean(tf.square(self.test_f(x, t)))
        loss = tf.reduce_mean(tf.square(self.obj_f(x, t)))
        # loss = tf.reduce_mean(tf.abs(y_-self.func_u(x, t)))

        # 计算L2正则化损失函数
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        # 计算模型的正则化损失。一般只计算神经网络边上的权重的正则化损失，而不使用偏置项
        regularization = regularizer(self.weights_out)
        for i in range(self.NUM_LAYER):
            regularization += regularizer(self.weights[i])

        # 总损失等于交叉熵损失+正则化损失
        loss_R = loss + regularization

        # 设置指数衰减的学习率
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            TRAINING_STEPS,  # 过完所有训练数据需要的迭代次数
            LEARNING_RATE_DECAY)

        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_R, global_step=global_step)
        #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_R, global_step=global_step)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        accuracy = tf.reduce_mean(tf.square(self.test_f(x, t)))

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # 准备验证数据。
        self.generate_data()
        valid_x, valid_t, y = self.get_valid_data()
        validate_feed = {x: valid_x, t: valid_t, y_: y}
        validate_acc = self.sess.run(accuracy, feed_dict=validate_feed)
        print(validate_acc)

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            train_x, train_t, y = self.get_train_data()
            train_dict = {x: train_x, t: train_t, y_: y}
            self.sess.run(train_step, feed_dict=train_dict)

            # 每100轮输出一次在验证数据集上的测试结果
            if i % 100 == 0:
                validate_acc = self.sess.run(accuracy, feed_dict=train_dict)
                print('After %d training steps, validation accuracy is %g' % (i, validate_acc))
                loss_value = self.sess.run([loss_R, loss], feed_dict=train_dict)
                print('loss=', loss_value)

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        # print(sess.run(self.obj_f(x,t),feed_dict=validate_feed))
        # print(sess.run(self.func_u(x,t),feed_dict=validate_feed))
        test_acc = self.sess.run(accuracy, feed_dict=train_dict)
        print('After %d training steps, test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))

    def plotting(self):
        fig = plt.figure()
        res = 100
        # ax = Axes3D(fig)
        # arange函数用于创建等差数组
        # X = np.arange(-1, 1, 0.02).astype('float32')
        # Y = np.arange(0, 1, 0.01).astype('float32')
        # numpy.linspace函数可以创建一个等差序列的数组，常用3个参数，起始值，终止值，序列长度
        X = np.linspace(-1, 1, res).astype('float32')
        Y = np.linspace(0, 1, res).astype('float32')
        # Z = np.random.uniform(0, 1, [100, 100])
        for i in range(res):
            x = np.array([X[i]] * res)
            x = x.reshape([res, 1])
            y = Y.reshape([res, 1])
            u = self.sess.run(self.func_u(x, y))
            f = self.sess.run(self.obj_f(x, y))
            # print('shape of u:', np.shape(u))
            # u=np.reshape(u,[100,1])
            # print(u)
            if i == 0:
                # Z=np.hstack(u)
                Z = u
                Z2 = np.abs(f)
            else:
                Z = np.hstack((Z, u))
                Z2 = np.hstack((Z2, np.abs(f)))
                # print('shape of u:', np.shape(u))
                # print('shape of Z:', np.shape(Z))
                # Z = np.concatenate((Z, u), 1)

        # print('shape of Z:', np.shape(Z))
        # print(np.shape(np.concatenate([self.func_u(x,y),self.func_u(x,y)],1)))
        X, Y = np.meshgrid(X, Y)

        # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='rainbow')

        Z3 = np.sin(np.pi * (X * X + Y * Y))
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot_surface(X, Y, Z3, rstride=1, cstride=1, cmap='rainbow')

        plt.show()


# 主程序入口
def main(argv=None):
    network = NN()
    network.train()
    network.plotting()


# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()

