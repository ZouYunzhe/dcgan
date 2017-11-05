import tensorflow as tf
from utils import *
import numpy as np

def demo_mnist():
    BATCH_SIZE = 64
    sample_labels_tmp = np.random.randint(0, 10, [BATCH_SIZE, 1])
    sample_labels = np.zeros(shape=[BATCH_SIZE, 10], dtype=np.float32)
    for i in range(BATCH_SIZE):
        sample_labels[i, sample_labels_tmp[i]] = 1
    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    sess = tf.Session()
    saver = tf.train.import_meta_graph("D:\MachineLearning\myGAN\logs\DCGAN_model.ckpt-1003.meta")
    saver.restore(sess, "D:\MachineLearning\myGAN\logs\DCGAN_model.ckpt-1003")
    print("Model Restored!")
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name('y:0')
    z = graph.get_tensor_by_name("z:0")
    feed_dict = {z: sample_z, y: sample_labels}
    G = graph.get_tensor_by_name("generate_image:0")
    sample = sess.run(G, feed_dict)
    samples_path = 'D:/MachineLearning/myGAN/images/'
    save_images(sample, [8, 8],
                samples_path + 'test2.png')
    print('save down')
    sess.close()


if __name__ == '__main__':
    demo_mnist()
