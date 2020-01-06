import tensorflow as tf
import numpy as np


# 变量
def variable_test():
    # 定义变量
    w = tf.Variable([[0.5, 1.0]])
    X = tf.Variable([[2.0], [2.2]])
    # 矩阵相乘
    y = tf.matmul(X, w)
    # 以上操作只是得到一个框架，没有赋值，赋值操作如下
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print(y.eval())

    ''' 数据基本操作 '''
    # 0、1值初始化
    init_zero = tf.zeros([3, 4], tf.float32)
    init_one = tf.ones([2, 3], tf.float32)
    # like 操作，用0、1值得到一个相同类型的矩阵
    zero_like = tf.zeros_like(X)
    one_like = tf.ones_like(w)
    # 将数据转换为 tensor 的格式
    tensor = tf.constant([1, 2, 3, 9, 5])
    # linspace 开始到结束创建几个数
    tensor_linspace = tf.linspace(3.0, 12.0, 4, name='linspace')
    # rang 开始到结束创建以多少为间隔的数
    tensor_rang = tf.range(3, 18, 3)
    # 创建均值为0方差为1的随机值
    norm = tf.random_normal([2, 3], mean=0, stddev=1)
    # 洗牌操作，打乱顺序
    tensor_order = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 1]])
    shuffle = tf.random_shuffle(tensor_order)
    # numpy 转换
    np_zero = np.zeros((3, 3))
    np_tf = tf.convert_to_tensor(np_zero)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('init_zero\n', init_zero.eval())
        print('init_one\n', init_one.eval())
        print('zero_like\n', zero_like.eval())
        print('one_like\n', one_like.eval())
        print('tensor\n', tensor.eval())
        print('tensor_linspace\n', tensor_linspace.eval())
        print('tensor_rang\n', tensor_rang.eval())
        print('norm\n', norm.eval())
        print('shuffle\n', shuffle.eval())
        print('np_zeros\n', np_zero)
        print('np_tf\n', np_tf.eval())
        save_path = tf.train.Saver().save(sess, './data/test')
        print(save_path)
    # 以0位变量的初始化值 对该值每次进行 +1操作
    zero_variable = tf.Variable(0)
    new_value = tf.add(zero_variable, tf.constant(1))
    update = tf.assign(zero_variable, new_value)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(3):
            sess.run(update)
        print('zero_change\n', zero_variable.eval())
        print('update\n', update.eval())
    # 新建框架再赋值
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    with tf.Session() as sess:
        placeholder = sess.run([output], feed_dict={input1: [7.0], input2: [2.0]})
        print('placeholder\n', placeholder)


if __name__ == '__main__':
    variable_test()
