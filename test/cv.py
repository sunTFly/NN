import cv2
import numpy as np

img = cv2.imread('./image/1.png', 0)
img = np.reshape(img, (1,-1))
print(img.shape)
# img = cv2.resize(img, (28, 28))
# imgc = img.copy()
# imgc[img > 50] = [0]
# imgc[img <= 50] = [255]
# cv2.imwrite('./image/3c.jpg', imgc)
