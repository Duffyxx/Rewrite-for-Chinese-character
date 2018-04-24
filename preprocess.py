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
from glob import glob

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


def process_font(chars, font_path, x_offset=50, y_offset=40):
    char_size = 150
    canvas = 256

    font_bitmaps = generate_font_bitmaps(chars, font_path, char_size,
                                         canvas, x_offset, y_offset)
    return font_bitmaps

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

def combine(font_1, font_2, type, output):
    if not os.path.exists(output):
        os.mkdir(output)
    print('combine: ', type)
    for i in range(len(font_1)):
        tmp = Image.fromarray(np.concatenate([font_1[i], font_2[i]], axis=1))
        tmp = Image.merge('RGB', (tmp, tmp, tmp))
        tmp.save(output+str(type)+'_'+str(i)+'.jpg', 'jpeg')
    print('Done')


if __name__ == "__main__":
    if not os.path.exists('./character/'):
        os.mkdir('./character/')
    font = glob('./font/*.TTF') + glob('./font/*.ttf')
    chars = get_chars_set('./top_3000_simplified.txt')
    data = []
    for i, file_name in enumerate(font):
        print('processing font : ' + file_name)
        data.append(process_font(chars, file_name))
    data = np.array(data)
    for i, each_data in enumerate(data[1:]):
        combine(data[0], each_data, i, './character/')
