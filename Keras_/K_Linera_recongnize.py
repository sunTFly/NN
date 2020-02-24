import keras
import numpy as np
import matplotlib.pyplot as plt
# 顺序模型
from keras.models import Sequential
# 全连接
from keras.layers import Dense

x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise

# 顺序模型
model = Sequential()
# 模型中添加全连接层
model.add(Dense(units=1, input_dim=1))
# sgd:随机梯度下降法，mse：均方误差
model.compile(optimizer='sgd', loss='mse')
for step in range(3001):
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
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()

