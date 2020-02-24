import keras
import numpy as np
import matplotlib.pyplot as plt
# 顺序模型
from keras.models import Sequential
# 全连接
from keras.layers import Dense
from keras.optimizers import SGD

x_data = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 顺序模型
model = Sequential()
# 模型中添加全连接层
model.add(Dense(units=10, input_dim=1,activation='relu'))
model.add(Dense(units=1,activation='relu'))
sgd = SGD(lr=0.3)
# sgd:随机梯度下降法，mse：均方误差
model.compile(optimizer=sgd, loss='mse')

for step in range(5001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print('cost:', cost)
# 打印权重和偏置值
W, b = model.layers[0].get_weights()
print('W:', W, 'b:', b)
# x_data输入网络中得到预测值
y_pred = model.predict(x_data)
# 显示随机点
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()
