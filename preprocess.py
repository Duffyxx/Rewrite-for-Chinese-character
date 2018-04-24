# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import os
import scipy.misc as misc
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

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


#reload(sys)
#sys.setdefaultencoding("utf-8")

FLAGS = None


def draw_char_bitmap(ch, font, char_size, x_offset, y_offset):
    image = Image.new("RGB", (char_size, char_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    gray = image.convert('L')
    bitmap = np.asarray(gray)
    return bitmap


def generate_font_bitmaps(chars, font_path, char_size, canvas_size, x_offset, y_offset):
    font_obj = ImageFont.truetype(font_path, char_size)
    bitmaps = list()
    for c in chars:
        bm = draw_char_bitmap(c, font_obj, canvas_size, x_offset, y_offset)
        bitmaps.append(bm)
    return np.array(bitmaps)


def process_font(chars, font_path, save_dir, x_offset=0, y_offset=0, mode='target'):
    char_size = 150
    canvas = 256
    #if mode == 'source':
    #    char_size *= 2
    #    canvas *= 2
    font_bitmaps = generate_font_bitmaps(chars, font_path, char_size,
                                         canvas, x_offset, y_offset)
    _, ext = os.path.splitext(font_path)
    if not ext.lower() in [".otf", ".ttf"]:
        raise RuntimeError("unknown font type found %s. only TrueType or OpenType is supported" % ext)
    _, tail = os.path.split(font_path)
    font_name = ".".join(tail.split(".")[:-1])
    bitmap_path = os.path.join(save_dir, "%s.npy" % font_name)
    np.save(bitmap_path, font_bitmaps)
    sample_image_path = os.path.join(save_dir, "%s_sample.png" % font_name)
    render_fonts_image(font_bitmaps[:100], sample_image_path, 10, False)
    print("%s font %s saved at %s" % (mode, font_name, bitmap_path))


def get_chars_set(path):
    """
    Expect a text file that each line is a char
    """
    chars = list()
    with open(path) as f:
        for line in f:
            line = u"%s" % line
            char = line.split()[0]
            chars.append(char)
    return chars

def combine(pathA, pathB, type, output):
    if not os.path.exists(output):
        os.mkdir(output)
    font_2 = np.load(pathA)
    font_1 = np.load(pathB)
    for i in range(len(font_1)):
        tmp = Image.fromarray(np.concatenate([font_1[i], font_2[i]], axis=1))
        tmp = Image.merge('RGB', (tmp, tmp, tmp))
        tmp.save(output+str(type)+'_'+str(i)+'.jpg', 'jpeg')
    print('Done')

def combine_npy(pathA, pathB, pathC, pathD):
    src = np.array(np.load(pathA))
    tgt = []
    tgt.append(np.load(pathB))
    tgt.append(np.load(pathC))
    tgt.append(np.load(pathD))
    data = []
    label = []
    for a in range(len(src)):
        for i in range(len(tgt)):
            tmp = np.concatenate([src[a], tgt[i][a]], axis=1)
            tmp = tmp.reshape([256, 512, 1])
            data.append(np.concatenate([tmp,tmp,tmp], axis=2))
            label.append(i)
    np.save('data.npy', data)
    np.save('label.npy', label)
    print('Done')


if __name__ == "__main__":
    if not os.path.exists('./bitmap/'):
        os.mkdir('./bitmap/')
    if not os.path.exists('./charactor/'):
        os.mkdir('./charactor/')
    chars = get_chars_set('./charsets/top_3000_simplified.txt')[:2100]
    process_font(chars, "./0.ttf", './bitmap/', 0, 0, mode='source')
    process_font(chars, "./1.ttf", './bitmap/', 0, 0, mode='source')
    process_font(chars, "./2.ttf", './bitmap/', 0, 0, mode='source')
    process_font(chars, "./3.ttf", './bitmap/', 0, 0, mode='source')
    combine('./bitmap/0.npy', './bitmap/0.npy', 0, './charactor/')
    combine('./bitmap/2.npy', './bitmap/2.npy', 2, './charactor/')
    combine('./bitmap/3.npy', './bitmap/3.npy', 3, './charactor/')
    combine('./bitmap/1.npy', './bitmap/1.npy', 1, './charactor/')
    # combine_npy('./bitmap/0.npy', './bitmap/1.npy', './bitmap/2.npy', './bitmap/3.npy')
