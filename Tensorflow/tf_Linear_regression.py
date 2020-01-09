import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X_data = []
y_data = []
# 初始化数据
for _ in range(2000):
    X = np.random.random()
    y = 0.1 * X + 0.5 + np.random.normal(0.0, 0.01)
    X_data.append(X)
    y_data.append(y)
# 初始化参数
W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros(1))
# 选择梯度下降方法进行训练
y_hat = X_data * W + b
loss = tf.reduce_mean(tf.square(y_hat - y_data))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
a = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        sess.run(train)
        print('w: ', W.eval(), '    b: ', b.eval(), '   loss: ', loss.eval())
    plt.scatter(X_data, y_data, c='r')
    plt.plot(X_data, sess.run(y_hat))
    plt.show()
