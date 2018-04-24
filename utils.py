# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import PIL.Image as im
import glob
import os

def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.get_variable('b', [output_filters], initializer=tf.constant_initializer(0.0))
        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b

def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon,
                                        scale=True, is_training=is_training, scope=scope)

def deconv2d(x, output_channels, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, output_channels, input_shape[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=[16, input_shape[1]*2, input_shape[2]*2, output_channels],
                                        strides=[1, sh, sw, 1])

        biases = tf.get_variable('b', [output_channels], initializer=tf.constant_initializer(0.0))
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fc(x, output_size, stddev=0.02, scope="fc"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, W) + b

def concat_label(x, label, duplicate=1):
    x_shape = x.get_shape().as_list()
    if duplicate < 1:
        return x
    # duplicate the label to enhance its effect, does it really affect the result?
    label = tf.tile(label, [1, duplicate])
    label_shape = label.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat([x, label],axis=1)
    elif len(x_shape) == 4:
        label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
        return tf.concat([x, label*tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])], axis=3)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img

class Batch_Data(object):
    def __init__(self, batch_size, input_dir):
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.iter = 0
        self.val_iter = 0
        self.load_images()
        self.train_count = len(self.train_data)
        self.val_count = len(self.val_data)
        self.train_batches = int(self.train_count/self.batch_size)
        self.val_batches = int(self.val_count/self.batch_size)
        print('data loaded, %d samples, and %d batches'%(self.train_count, self.train_batches))

    def load_images(self):
        # input_paths = glob.glob(os.path.join(self.input_dir, "*.jpg"))
        data = np.load('data.npy')
        labels = np.load('label.npy')
        data = data/127.5-1
        # for image in input_paths:
        #     print(image)
        #     data.append(np.array(im.open(image))/127.5 - 1)
        #     labels.append(int(image.split('_')[-1].split('.')[0]))
        self.train_data = np.array(data)[0:8000]
        self.train_labels = np.array(labels)[0:8000]
        self.val_data = np.array(data)[8000:]
        self.val_labels = np.array(labels)[8000:]

    def next_batch(self):
        data = self.train_data[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
        label = self.train_labels[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
        self.iter += 1
        if self.iter == self.train_batches - 1:
            self.iter = 0
        return label, data

    def next_val_batch(self):
        data = self.val_data[self.val_iter*self.batch_size:(self.val_iter+1)*self.batch_size]
        label = self.val_labels[self.val_iter*self.batch_size:(self.val_iter+1)*self.batch_size]
        self.val_iter += 1
        if self.val_iter == self.val_batches - 1:
            self.val_iter = 0
        return label, data
