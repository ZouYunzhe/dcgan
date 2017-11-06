# -*- coding: utf-8 -*-
import tensorflow as tf
import os

from read_mnist_data import *
from utils import *
from ops import *
from model import *
from model import BATCH_SIZE

def read_and_decode(tfrecords_file,batch_size):
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
    image = tf.reshape(image, [28, 28, 1])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)
    depth = 10
    return image_batch, tf.one_hot(tf.reshape(label_batch, [batch_size]), depth)


def train():
    # 设置 global_step ，用来记录训练过程中的 step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 训练过程中的日志保存文件
    train_dir = '/home/zouyunzhe/dc/non_logs'
    tfrecords_file = '/home/zouyunzhe/dc/data_notmnist/notMNIST/test.tfrecords'
    with tf.name_scope('input'):
        data_x, data_y = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='y')
    images = tf.placeholder(tf.float32, [64, 28, 28,1], name='real_images')
    z = tf.placeholder(tf.float32, [None, 100], name='z')

    # 由生成器生成图像 G
    G = generator(z, y)
    # 真实图像送入判别器
    D, D_logits = discriminator(images, y)
    # 采样器采样图像
    samples = sampler(z, y)
    # 生成图像送入判别器
    D_, D_logits_ = discriminator(G, y, reuse=True)

    # 损失计算
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

    # # 总结操作
    # z_sum = tf.summary.histogram ("z", z)
    # d_sum = tf.summary.histogram("d", D)
    # d__sum = tf.summary.histogram("d_", D_)
    # G_sum = tf.summary.image("G", G)
    #
    # d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    # d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    # d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    # g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    #
    # # 合并各自的总结
    # g_sum = tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
    # d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    # 生成器和判别器要更新的变量，用于 tf.train.Optimizer 的 var_list
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()

    # 优化算法采用 Adam
    d_optim = tf.train.GradientDescentOptimizer(0.0002) \
        .minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.GradientDescentOptimizer(0.0002) \
        .minimize(g_loss, var_list=g_vars, global_step=global_step)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(train_dir, sess.graph)



    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    tmp = np.linspace(0, 9, 10)
    sample_labels_tmp = np.tile(tmp, (1, 7))
    sample_labels = np.zeros(shape=[BATCH_SIZE, 10], dtype=np.float32)
    for k in range(BATCH_SIZE):
        sample_labels[k, int(sample_labels_tmp[0, k])] = 1
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
    # 循环 25 个 epoch 训练网络
        for epoch in range(25):
            batch_idxs = 1000
            for idx in range(batch_idxs):
                if coord.should_stop():
                    break
                batch_images, batch_labels = sess.run([data_x, data_y])
                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

                # 更新 D 的参数
                sess.run([d_optim], feed_dict={images: batch_images, z: batch_z, y: batch_labels})
                # writer.add_summary(summary_str, idx + 1)

                # 更新 G 的参数
                sess.run([g_optim], feed_dict={z: batch_z, y: batch_labels})
                # writer.add_summary(summary_str, idx + 1)

                # 更新两次 G 的参数确保网络的稳定
                sess.run([g_optim], feed_dict={z: batch_z, y: batch_labels})
                # writer.add_summary(summary_str, idx + 1)

                # 计算训练过程中的损失，打印出来
                errD_fake = d_loss_fake.eval(feed_dict={z: batch_z, y: batch_labels}, session=sess)
                errD_real = d_loss_real.eval(feed_dict={images: batch_images, y: batch_labels}, session=sess)
                errG = g_loss.eval(feed_dict={z: batch_z, y: batch_labels}, session=sess)

                if idx % 20 == 0:
                    print("Epoch: [%2d] [%4d/%4d] d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, errD_fake + errD_real, errG))

                # 训练过程中，用采样器采样，并且保存采样的图片到
                # /home/your_name/TensorFlow/DCGAN/samples/
                if idx == 999:
                    sample = sess.run(samples, feed_dict={z: sample_z, y: sample_labels})
                    samples_path = '/home/zouyunzhe/dc/non_samples/'
                    save_images(sample, [8, 8],
                                samples_path + 'test_%d_epoch_%d.png' % (epoch, idx))
                    print('save down')
        if epoch == 24:
            checkpoint_path = os.path.join(train_dir, 'DCGAN_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    train()