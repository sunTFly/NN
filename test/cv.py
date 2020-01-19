import cv2
import numpy as np
import tensorflow as tf

# img = cv2.imread('./image/1.png', 0)
# img = np.reshape(img, (1, -1))
# print(img.shape)
# img = cv2.resize(img, (28, 28))
# imgc = img.copy()
# imgc[img > 50] = [0]
# imgc[img <= 50] = [255]
# cv2.imwrite('./image/3c.jpg', imgc)
#
# a = tf.Variable(tf.random_normal([5, 49, 49, 3]))
# b = [7 * 7, 3]
# c = tf.reshape(a, [-1, 7*7])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(tf.shape(c).eval())
