# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import argparse
from model.unet import UNet
from model.utils import compile_frames_to_gif
import scipy.misc as misc
import numpy as np

def render_fonts_image(x, path, img_per_row, unit_scale=True):
    if unit_scale:
        # scale 0-1 matrix back to gray scale bitmaps
        bitmaps = (x * 255.).astype(dtype=np.int16) % 256
    else:
        bitmaps = x
    num_imgs, w, h = x.shape
    assert w == h
    side = int(w)
    width = img_per_row * side
    height = int(np.ceil(float(num_imgs) / img_per_row)) * side
    canvas = np.zeros(shape=(height, width), dtype=np.int16)
    # make the canvas all white
    canvas.fill(255)
    for idx, bm in enumerate(bitmaps):
        x = side * int(idx / img_per_row)
        y = side * int(idx % img_per_row)
        canvas[x: x + side, y: y + side] = bm
    misc.toimage(canvas).save(path)
    return path

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if not os.path.exists('./out/'):
        os.mkdir('./out/')
    with tf.Session(config=config) as sess:
        model = UNet(batch_size=16, is_train=False)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=False)
        fake_imgs = model.infer_code(model_dir='./exp/checkpoint/experiment_0_batch_16/')
        fake_imgs = np.array(fake_imgs)
        for i in range(16):
            print('generate images: ', i)
            fake = fake_imgs[:,i][:,:,:,0]
            render_fonts_image(fake, './out/'+ str(i)+ '.png', 10)
    print('finished.')
if __name__ == '__main__':
    tf.app.run()
