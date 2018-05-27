__author__ = 'qcp'
# -*- coding:utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LEARNING_RATE_BASE = 0.001  # 学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 1000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


class NN:
    def __init__(self):

        # 数据集相关参数
        self.BATCH_SIZE = 100  # 一个训练batch中的训练数据个数

        # 神经网络相关参数
        self.LAYER_NODE = [1, 50, 50,50, 1]
        self.NUM_LAYER = len(self.LAYER_NODE)  # 隐藏层数
        # 生成隐藏层的参数
        self.initializer = tf.contrib.layers.xavier_initializer()
        # self.initializer = tf.truncated_normal_initializer
        self.weights = []
        self.biases = []
        for i in range(self.NUM_LAYER - 1):
            # self.weights.append(
            #    tf.Variable(tf.truncated_normal(shape=[self.LAYER_NODE[i], self.LAYER_NODE[i+1]],stddev=0.1)))
            # self.biases.append(tf.Variable(tf.constant(0.1,shape=[self.LAYER_NODE[i+1]])))
            self.weights.append(tf.get_variable('weight' + str(i), shape=[self.LAYER_NODE[i], self.LAYER_NODE[i + 1]],
                                                initializer=self.initializer))
            self.biases.append(tf.Variable(tf.constant(0.1, shape=[self.LAYER_NODE[i + 1]])))

        self.init_op1 = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init_op1)

    def inference(self, input_tensor):
        '''辅助函数，给定网络的输入和所有参数，计算神经网络的前向传播结果。'''

        # 计算输出层前向传播结果。
        layer = []
        layer.append(tf.nn.relu(tf.matmul(input_tensor, self.weights[0]) + self.biases[0]))
        for i in range(1, self.NUM_LAYER - 2):
            layer.append(tf.nn.relu(tf.matmul(layer[i - 1], self.weights[i]) + self.biases[i]))

        return tf.matmul(layer[-1], self.weights[-1]) + self.biases[-1]

    # u函数值
    def func_u(self, x):
        u = self.inference(x)
        return u

    # f函数值
    def obj_f(self, x):
        u = self.func_u(x)
        f = u - tf.square(x)
        return f

    def get_train_data(self):
        # 生成训练点
        # x = np.random.uniform(-1, 1, [self.BATCH_SIZE])
        x = np.linspace(-1, 1, self.BATCH_SIZE)
        train_x = x.reshape([self.BATCH_SIZE, 1])
        y = train_x * train_x
        return train_x, y

    def get_valid_data(self):
        # 验证点
        a = np.loadtxt('axis1.txt')
        x = []
        for i in range(self.BATCH_SIZE):
            x.append(a[i])
        x = np.array(x).reshape([self.BATCH_SIZE, 1])
        y = x * x
        return x, y

    def generate_data(self):
        f = open('axis1.txt', 'w')
        x = np.random.uniform(-1, 1, [self.BATCH_SIZE])
        for i in range(self.BATCH_SIZE):
            f.write(str(x[i]) + '\n')
        f.close()

    # 训练模型的过程
    def train(self):
        # 输入节点
        x = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y-input')

        # 定义存储训练轮数的变量，
        # 不需要计算滑动平均值，所以指定为不可训练量。
        # 在使用TensorFlow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
        global_step = tf.Variable(0, trainable=False)

        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.obj_f(x))))
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.func_u(x) - y_)))

        # 计算L2正则化损失函数
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        # 计算模型的正则化损失。一般只计算神经网络边上的权重的正则化损失，而不使用偏置项
        regularization = 0
        for i in range(self.NUM_LAYER - 1):
            regularization += regularizer(self.weights[i])

        loss_R = loss + regularization
        # 设置指数衰减的学习率
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            TRAINING_STEPS,  # 过完所有训练数据需要的迭代次数
            LEARNING_RATE_DECAY)
        learning_rate = LEARNING_RATE_BASE

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_R, global_step=global_step)

        accuracy = tf.reduce_mean(tf.square(self.obj_f(x)))

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # 准备验证数据。
            self.generate_data()
            valid_x, y = self.get_valid_data()
            validate_feed = {x: valid_x, y_: y}
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            print('Before training, accuracy = ', validate_acc)

            # 迭代训练神经网络
            for i in range(TRAINING_STEPS):
                # 每100轮输出一次在验证数据集上的测试结果
                if i % 100 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %d training steps, validation accuracy is %g' % (i, validate_acc))
                    loss_value = sess.run([loss_R, loss], feed_dict=validate_feed)
                    print('loss_R, loss = ', loss_value)
                # 产生这一轮使用的一个batch的训练数据，并运行训练过程
                train_x, y = self.get_train_data()
                train_dict = {x: train_x, y_: y}
                sess.run(train_step, feed_dict=train_dict)

            test_acc = sess.run(accuracy, feed_dict=train_dict)
            print('After %d training steps, test accuracy is %g' % (TRAINING_STEPS, test_acc))

    def plotting(self):
        fig = plt.figure()
        # ax = Axes3D(fig)
        # arange函数用于创建等差数组
        # np.linspace()也可以
        x = np.arange(-1, 1, 0.02).astype('float32')
        x = x.reshape([100, 1])
        y = self.sess.run(self.func_u(x) - self.obj_f(x))
        u = self.sess.run(self.func_u(x))
        plt.plot(x, y, color='r', linewidth=2)
        plt.plot(x, u, color='b', linewidth=2)
        plt.show()


# 主程序入口
def main(argv=None):
    np.random.seed(1000)
    network = NN()
    network.train()
    network.plotting()


# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
