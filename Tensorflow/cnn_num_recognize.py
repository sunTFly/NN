import tensorflow as tf
import numpy as np
import input_data

''' 获取数据集 '''

mnist = input_data.read_data_sets('./data', one_hot=True)
train_data = mnist.train.images
train_lable = mnist.train.labels
test_data = mnist.test.images
test_lable = mnist.test.labels
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 随机数
keepProb = tf.placeholder(tf.float32)

''' 
初始化参数
    Weights:
        Wc[n1,n2,n3,n4]：卷积层参数
            n1,n2:卷积核的长和宽；n3:图像的深度；n4：卷积核的个数
        Wd[n1,n2]：全连接层参数
            n1:输入；n2:输出
    biases：偏置项
 '''
Weights = {
    'Wc1': tf.Variable(tf.random_uniform([3, 3, 1, 64], -1, 1)),
    'Wc2': tf.Variable(tf.random_uniform([3, 3, 64, 128], -1, 1)),
    'Wd1': tf.Variable(tf.random_uniform([7 * 7 * 128, 1024], -1, 1)),
    'Wd2': tf.Variable(tf.random_uniform([1024, 10], -1, 1))
}
biases = {
    'bc1': tf.Variable(tf.zeros([64])),
    'bc2': tf.Variable(tf.zeros([128])),
    'bd1': tf.Variable(tf.zeros([1024])),
    'bd2': tf.Variable(tf.zeros([10]))
}

''' 卷积神经网络前向传播 '''


def convolution_forward(_input, _Weights, _biases, _keepProb):
    input_shape = tf.reshape(_input, [-1, 28, 28, 1])
    # tf.nn.conv2d 卷积： 参数说明,第一个是四维的输入数据；第二个是卷积核；
    # 第三个是步长，四个维度分别是批量数据、长、宽、深度的步长
    # padding有两个选择 SAME：代表有边界0填充，VALID：代表无边界填充
    c1 = tf.nn.conv2d(input_shape, _Weights['Wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数
    c1 = tf.nn.relu(tf.nn.bias_add(c1, _biases['bc1']))
    # tf.nn.max_pool 最大值池化: 参数和卷积类似，第二个参数为池化的大小
    p1 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第二层
    c2 = tf.nn.conv2d(p1, _Weights['Wc2'], strides=[1, 1, 1, 1], padding='SAME')
    c2 = tf.nn.relu(tf.nn.bias_add(c2, _biases['bc2']))
    p2 = tf.nn.max_pool(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 全连接层
    # 随机扔掉一部分神经元，防止过拟合
    p2 = tf.nn.dropout(p2, _keepProb)
    f_shape = tf.reshape(p2, [-1, 7 * 7 * 128])
    f1 = tf.nn.relu(tf.add(tf.matmul(f_shape, _Weights['Wd1']), _biases['bd1']))
    f1 = tf.nn.dropout(f1, _keepProb)
    f_out = tf.add(tf.matmul(f1, _Weights['Wd2']), _biases['bd2'])
    return f_out


''' 选择一种方法进行计算 '''
result = convolution_forward(x, Weights, biases, keepProb)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
# 训练集准确率
_corr = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))

''' 进行迭代 '''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        avg_loss = 0.0
        for i in range(10):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keepProb: 0.6})
            avg_loss += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keepProb: 1.0}) / 10
        train_accr = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepProb: 1.0})
        print('训练集准确率', train_accr)
