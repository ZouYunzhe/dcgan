# -*- coding: utf-8 -*-
import tensorflow as tf
import os

from read_mnist_data import *
from utils import *
# from ops import *
# from model import *
from model import BATCH_SIZE
import numpy as np


# with tf.Session() as sess:
#     # 通过张量的名称来获取张量
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))


def demo_mnist():
    BATCH_SIZE = 64
    # y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='y')
    # z = tf.placeholder(tf.float32, [None, 100], name='z')
    sample_labels_tmp = np.random.randint(0, 10, [BATCH_SIZE, 1])
    sample_labels = np.zeros(shape=[BATCH_SIZE, 10], dtype=np.float32)
    for i in range(BATCH_SIZE):
        sample_labels[i, sample_labels_tmp[i]] = 1
    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

    # G = generator(z, y)


    # 生成器和判别器要更新的变量，用于 tf.train.Optimizer 的 var_list
    # t_vars = tf.trainable_variables()
    # aa=1
    # d_vars = [var for var in t_vars if 'd_' in var.name]
    # g_vars = [var for var in t_vars if 'g_' in var.name]

    sess = tf.Session()

    # restore model
    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph("D:\MachineLearning\myGAN\logs\DCGAN_model.ckpt-1003.meta")
    saver.restore(sess, "D:\MachineLearning\myGAN\logs\DCGAN_model.ckpt-1003")
    print("Model Restored!")
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name('y:0')
    z = graph.get_tensor_by_name("z:0")
    feed_dict = {z: sample_z, y: sample_labels}
    G = graph.get_tensor_by_name("generate_image:0")
    # print(sess.run(G, feed_dict))
    #
    # samples = sampler(z, y)
    sample = sess.run(G, feed_dict)
    samples_path = 'D:/MachineLearning/myGAN/images/'
    save_images(sample, [8, 8],
                samples_path + 'test2.png')
    print('save down')
    sess.close()


if __name__ == '__main__':
    demo_mnist()
