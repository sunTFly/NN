import tensorflow as tf
import scipy.io
import numpy as np
import cv2

''' 减均值 '''


def preprocess(image, mean_pixel):
    return image - mean_pixel


''' 卷积层操作
        调用 tf.nn.conv2d 方法
 '''


def conv_layer(input_img, kernels, bias):
    conv = tf.nn.conv2d(input_img, tf.constant(kernels), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


'''  池化层操作 '''


def pool_layer(input_img):
    return tf.nn.max_pool(input_img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


''' vgg_19 '''


def vgg_net():
    data = scipy.io.loadmat('./save/imagenet-vgg-verydeep-19.mat')

    d = data["classes"]
    lables = d[0][0][0][0]
    descriptions = d[0][0][1][0]
    img = cv2.imread('./data/image/cat1.jpg', 1)
    img = cv2.resize(img, (224, 224))
    image = tf.placeholder(tf.float32)
    # 参数
    weights = data['layers'][0]
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    # 减均值
    input_image = np.array([preprocess(img, mean_pixel)])
    # VGG_19的各层 详情查看 './data/vgg_19.svg'
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    )
    net = {}
    feature = image
    for i, layer in enumerate(layers):
        layer_4 = layer[0:4]
        if layer_4 == 'conv':
            # kernels 卷积核；bias 偏置项
            kernels, bias = weights[i][0][0][0][0]
            # 由于原 kernels 为 宽、长、图像深度、卷积核个数，
            # 而 TensorFlow 需要的是 是 长、宽、图像深度、卷积核个数，所有通过 np.transpose 进行转换
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            feature = conv_layer(feature, kernels, bias)
        elif layer_4 == 'relu':
            feature = tf.nn.relu(feature)
        elif layer_4 == 'pool':
            feature = pool_layer(feature)
        net[layer] = feature
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        features = net['conv1_1'].eval(feed_dict={image: input_image})
        cv2.imshow('a', features[0, :, :, 0:3])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


vgg_net()
