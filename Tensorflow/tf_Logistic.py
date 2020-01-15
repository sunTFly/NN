import input_data
import cv2
import numpy as np
import tensorflow as tf

# 训练数据源
mnist = input_data.read_data_sets('./data', one_hot=True)
train_data = mnist.train.images
train_lable = mnist.train.labels
test_data = mnist.test.images
test_lable = mnist.test.labels

# 先创建一个存储数据的框架
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
# 初始化参数 W 和 b
W = tf.Variable(tf.random_uniform([784, 10], -1, 1))
b = tf.Variable(tf.zeros([10]))
# 选择 softmax 分类器
h_thta = tf.nn.softmax(tf.matmul(x, W) + b)
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h_thta), reduction_indices=1))
# 选择梯度下降方法
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
img = cv2.imread('./data/image/2c.jpg', 0)
img = tf.cast(img, tf.float32)
img_num = tf.reshape(img, (1, -1))
print(img_num.shape)
# 给框架赋值并初始化其它数据，用批量梯度下降方法进行迭代
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        n = int(mnist.train.num_examples / 100)
        for _ in range(n):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    print('W:', W.eval())
    print('===')
    print('b:', b.eval())
    h_thta_test = tf.nn.softmax(tf.matmul(img_num, W) + b)
    sess.run(h_thta_test)
    print('---')
    print(h_thta_test.eval())
