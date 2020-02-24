import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import input_data

mnist = input_data.read_data_sets('./data', one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
print(x_train.shape, y_train.shape)
# 创建模型 输入784个神经元，输出10个神经元
model = Sequential([Dense(units=10, input_dim=784, activation='relu')])
# 定义优化器
sgd = SGD(lr=0.2)
# 定义优化器 loss function ，训练过程中计算准确率
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)
# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('loss', loss)
print('acc', accuracy)
