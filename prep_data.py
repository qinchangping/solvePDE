__author__ = 'qcp'
# -*- coding:utf-8 -*-
import numpy as np

import tensorflow as tf

if 0:
    f = open('axis.txt', 'w')
    x = np.random.uniform(-1, 1, [100])
    index = np.random.randint(0, 100, [4])
    x[index[0]] = -1
    x[index[1]] = 1
    t = np.random.uniform(0, 1, [100])
    t[index[2]] = 0
    t[index[3]] = 1
    for i in range(100):
        f.write(str(x[i]) + ' ' + str(t[i]) + '\n')
    f.close()

if 0:
    a = np.loadtxt('axis.txt')
    # print(a)
    print(a.shape)
    x = []
    for i in range(100):
        x.append(a[i][1])
    x = np.array(x).reshape([100, 1])
    print(x)

# print(type(a))

'''from ast import literal_eval
with open("axis.txt") as f:
    c=f.readlines()
    lst = literal_eval(c)
    print(lst)
    print(type(lst))'''

'''
def get_train_data(batch_size):
    # 生成训练点
    train_x = np.random.uniform(-1, 1, [batch_size, 1])
    train_x[0] = -1
    train_x[-1] = 1
    train_t = np.random.uniform(0, 1, [batch_size, 1])
    train_t[0] = 0
    train_t[-1] = 1
    return train_x, train_t


num_batch = 200
batch_size = 100


filename = 'axis.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)
for i in range(num_batch):
    x, t = get_train_data(batch_size)

    image_raw = images[index].tostring()
    # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    # 将一个Example写入TFRecord文件
    writer.write(example.SerializeToString())
writer.close()'''

# tf.gradient()输入的函数必须是已经初始化的有数值的变量，如果是空变量会报错
# array连接不能直接连接，应该用tf.concat(),否则tf.gradient报错
if 0:
    x1 = np.random.uniform(-1.0, 1.0, [5, 1])
    x2 = np.random.uniform(0.0, 1.0, [5, 1])
    x = tf.Variable(x1)
    t = tf.Variable(np.random.uniform(0.0, 1.0, [5, 1]))
    a = tf.Variable([1.0, 2.0, 3.0])
    b = tf.Variable([3.0, 4.0, 5.0])
    c = tf.concat([x, t], 1) ** 2
    print(np.shape(c))
    grad = tf.gradients(c, [x, t])
    # grad1 = tf.gradients(grad, [x, t])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # g = sess.run(grad)
        # g1 = sess.run(grad1)
        # print(g)
        # print(g1)
        # sess.run(grad1)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if 0:
    def plotting():
        fig = plt.figure()
        # ax = Axes3D(fig)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        # arange函数用于创建等差数组
        X = np.arange(-1, 1, 0.02).astype('float32')
        Y = np.arange(0, 1, 0.01).astype('float32')
        Z1 = np.random.uniform(0, 1, [100, 100])
        # Z2=np.sin(np.pi*(Y))
        '''
        for i in range(100):
            x = np.array([X[i]] * 100)
            x = x.reshape([100, 1])
            y = Y.reshape([100, 1])
            u = self.sess.run(self.func_u(x, y))
            # print('shape of u:', np.shape(u))
            # u=np.reshape(u,[100,1])
            # print(u)
            if i == 0:
                # Z=np.hstack(u)
                Z = u
            else:
                Z = np.hstack((Z, u))
                # print('shape of u:', np.shape(u))
                # print('shape of Z:', np.shape(Z))
                # Z = np.concatenate((Z, u), 1)'''

        # print('shape of Z:', np.shape(Z))
        # print(np.shape(np.concatenate([self.func_u(x,y),self.func_u(x,y)],1)))
        X, Y = np.meshgrid(X, Y)

        # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
        ax1.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='rainbow')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        Z2 = np.sin(np.pi * (X * X + Y * Y))
        ax2.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='rainbow')

        plt.show()


    plotting()
if 0:
    BATCH_SIZE = 10000
    x = np.random.uniform(-1, 1, [BATCH_SIZE])
    t = np.random.uniform(0, 1, [BATCH_SIZE])
    y = np.sin(np.pi * (x * x + t * t))
    accuracy = tf.reduce_mean(tf.square(y))
    with tf.Session() as sess:
        a = sess.run(accuracy)
        print('a=', a)
if 1:
    import tensorflow as tf
    import numpy as np
    import math, random
    import matplotlib.pyplot as plt

    np.random.seed(1000) # for repro
    function_to_learn = lambda x: np.sin(x) + 0.1*np.random.randn(*x.shape)
    NUM_HIDDEN_NODES = 20
    NUM_EXAMPLES = 1000
    TRAIN_SPLIT = .8
    MINI_BATCH_SIZE = 100
    NUM_EPOCHS = 1000

    all_x = np.float32(
        np.random.uniform(-2*math.pi, 2*math.pi, (1, NUM_EXAMPLES))).T
    np.random.shuffle(all_x)
    train_size = int(NUM_EXAMPLES*TRAIN_SPLIT)
    trainx = all_x[:train_size]
    validx = all_x[train_size:]
    trainy = function_to_learn(trainx)
    validy = function_to_learn(validx)

    plt.figure(1)
    plt.scatter(trainx, trainy, c='green', label='train')
    plt.scatter(validx, validy, c='red', label='validation')
    plt.legend()



    X = tf.placeholder(tf.float32, [None, 1], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")

    def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
        if init_method == 'zeros':
            return tf.Variable(tf.zeros(shape, dtype=tf.float32))
        elif init_method == 'uniform':
            return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
        else: #xavier
            (fan_in, fan_out) = xavier_params
            low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
            high = 4*np.sqrt(6.0/(fan_in + fan_out))
            return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

    def model(X, num_hidden=10):
        w_h = init_weights([1, num_hidden], 'xavier', xavier_params=(1, num_hidden))
        b_h = init_weights([1, num_hidden], 'zeros')
        h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)

        w_o = init_weights([num_hidden, 1], 'xavier', xavier_params=(num_hidden, 1))
        b_o = init_weights([1, 1], 'zeros')
        return tf.matmul(h, w_o) + b_o

    yhat = model(X, NUM_HIDDEN_NODES)

    train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(yhat - Y))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    errors = []
    for i in range(NUM_EPOCHS):
        for start, end in zip(range(0, len(trainx), MINI_BATCH_SIZE), range(MINI_BATCH_SIZE, len(trainx), MINI_BATCH_SIZE)):
            sess.run(train_op, feed_dict={X: trainx[start:end], Y: trainy[start:end]})
        mse = sess.run(tf.nn.l2_loss(yhat - validy),  feed_dict={X:validx})
        errors.append(mse)
        if i%100 == 0: print( "epoch %d, validation MSE %g" % (i, mse))
    plt.plot(errors)
    plt.xlabel('#epochs')
    plt.ylabel('MSE')
    plt.show()