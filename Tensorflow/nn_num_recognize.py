import tensorflow as tf
import numpy as np
import input_data
import cv2

''' 手写测试，暂时无用 '''
img = cv2.imread('./data/image/2c.jpg', 0)
img = tf.cast(img, tf.float32)
img_num = tf.reshape(img, (1, -1))
print(img_num.shape)

''' 获取数据集 '''

mnist = input_data.read_data_sets('./data', one_hot=True)
train_data = mnist.train.images
train_lable = mnist.train.labels
test_data = mnist.test.images
test_lable = mnist.test.labels
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

''' 初始化参数 '''

Weights = {
    'W1': tf.Variable(tf.random_uniform([784, 256], -1, 1)),
    'W2': tf.Variable(tf.random_uniform([256, 128], -1, 1)),
    'W_out': tf.Variable(tf.random_uniform([128, 10], -1, 1)),
}
biases = {
    'b1': tf.Variable(tf.zeros([256])),
    'b2': tf.Variable(tf.zeros([128])),
    'b_out': tf.Variable(tf.zeros([10])),
}

''' 神经网络前向传播 '''


def forward_propagation(_x, _Weights, _biases):
    l1 = tf.nn.relu(tf.matmul(_x, _Weights['W1']) + _biases['b1'])
    l2 = tf.nn.relu(tf.matmul(l1, _Weights['W2']) + _biases['b2'])
    l_out = tf.matmul(l2, _Weights['W_out']) + _biases['b_out']
    return l_out


''' 
选择一种方法进行计算
    tf.nn.softmax_cross_entropy_with_logits(result, y)
        第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，
        它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
        第二个参数labels：实际的标签，大小同上
        第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，
        对于单样本而言，输出就是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）
        第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵
 '''
result = forward_propagation(x, Weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

''' 进行迭代 '''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    avg_loss = 0.0
    for _ in range(50):
        n = int(mnist.train.num_examples / 100)
        for _ in range(n):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_loss += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
        avg_loss = avg_loss / n
        print(avg_loss)
    h_thta_test = tf.nn.softmax(forward_propagation(img_num, Weights, biases))
    sess.run(h_thta_test)
    print(h_thta_test.eval())
