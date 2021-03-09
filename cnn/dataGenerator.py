#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: dataGenerator.py
# Created on: 2020/10/16

import random
import os
import pathlib
import numpy as np
import cv2
import time
from PIL import Image, ImageFont, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt


FONT_DIR = "../fonts/"
WORDS_DIR = "../words/"
OUTPUT_DIR = "../../DeepLearning/Datasets/captcha_CN/"  # 在此处修改样本生成文件夹
#OUTPUT_DIR = "validations/"


class captchaGenerator:
    def __init__(self, color=(0,0,0), size=(200, 50),  # color设置为“random”或R,G，B
                 fontSize=40):
        global FONT_DIR, WORDS_DIR
        font_dir = pathlib.Path(FONT_DIR)
        word_dir = pathlib.Path(WORDS_DIR)
        font_list = [str(path) for path in font_dir.glob("*.ttf")]
        word_list = [str(path) for path in word_dir.glob("*.txt")]

        print("loaded fonts: {}\nloaded words: {}".format(font_list,
                                                          word_list))

        words = []
        for i in word_list:
            f = open(i,"rb")
            t = f.read().decode(encoding="gbk").split(" ")
            words = words + t
            f.close()


        self.fontlist = font_list
        self.color = color
        self.size = size
        self.words = words
        self.wordnum = 1            # 暂时固定
        self.fontsize = fontSize


    def _rand_char(self):
        char_str = ""
        char_index = []
        for i in range(self.wordnum):
            index = random.randint(0,len(self.words))
            char_str += self.words[index]
            char_index.append(index)
        return char_str, char_index


    def _rand_color(self):
        return (random.randint(0, 200),
                random.randint(0, 200),
                random.randint(0, 200))


    def _rand_lineSpot(self, mode=0):
        width, height = self.size
        if mode == 0:
            return (random.randint(0, width), random.randint(0, height))
        elif mode == 1:
            return (random.randint(0, 6), random.randint(0, height))
        elif mode == 2:
            return (random.randint(width - 6, width), random.randint(0, height))


    def _rand_outerSpot(self):
        width = self.size[0]*4
        height = self.size[1]*4
        return (random.randint(0, width), random.randint(0, height))


    def generate(self, lineNum=48, wordIndex=None):
        width, height = self.size

        # 生成背景
        img = Image.new("RGB", (self.size[0]*4, self.size[1]*4),
                        (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 绘制杂线
        for i in range(lineNum//2):
            draw.line([self._rand_outerSpot(), self._rand_outerSpot()],
                      self._rand_color())
            draw.arc([self._rand_outerSpot(), self._rand_outerSpot()],
                     0,
                     360,
                     fill=self._rand_color(),
                     width=random.randint(1, 3))

        # 绘制字符
        if wordIndex == None:
            words, indexs = self._rand_char()
        else:
            words = [self.words[wordIndex]]
            indexs = [wordIndex]

        for i in range(len(words)):
            x = random.randint(0, width - self.fontsize*1.15)
            y = random.randint(0, height - self.fontsize*1.2)

            wordMap = Image.new("L",
                                (self.fontsize, self.fontsize))
            wordDraw = ImageDraw.Draw(wordMap)

            wordDraw.text((0, 0), words[i],
                          fill=255,
                          font=ImageFont.truetype(
                               self.fontlist[
                               random.randint(0, len(self.fontlist)-1)],
                          self.fontsize))
            rotated_map = wordMap.rotate(-30 + 60 * random.random(), expand=1)

            boxX = random.randint(0, self.size[0] * 3)
            boxY = random.randint(0, self.size[1] * 3)

            croped_img = img.crop((boxX, boxY,
                                   boxX + width, boxY + height))



            croped_img.paste(ImageOps.colorize(rotated_map,
                                        (0, 0, 0),
                                        self._rand_color()),
                      (x, y),
                      rotated_map)
            xmin = x
            ymin = y
            xmax = x + rotated_map.size[0]
            ymax = y + rotated_map.size[1]
            if xmax > self.size[0]:
                xmax = self.size[0]
            if ymax > self.size[1]:
                ymax = self.size[1]

        return np.array(croped_img), indexs, (xmin, ymin, xmax, ymax)


def path_fixer(path): # path checker
    chk = ""
    for i in path:
        chk += i
        if i in ("/", "\\"):
            if not os.path.exists(chk):
                os.mkdir(chk)


def position_check(data=None, label=None):
    if data is None or label is None:
        cg = captchaGenerator()
        cap = cg.generate()
        data = cap[0]
        label = cap[2]
    img = data
    xmin = label[0]
    ymin = label[1]
    xmax = label[2]
    ymax = label[3]
    print("position: ({},\t{})\t({},\t{})".format(
        xmin, ymin, xmax, ymax
    ))
    plt.imshow(img)
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                     fill=False, color="green")
    ax = plt.gca()
    ax.axes.add_patch(rect)
    plt.show()


def init():
    # 检查路径安全性
    paths = [FONT_DIR, WORDS_DIR, OUTPUT_DIR]
    for i in paths:
        path_fixer(i)


def main():
    cg = captchaGenerator()
    for i in range(5):
        res = cg.generate(48)
        print("now showing: {}".format(cg.words[res[1][0]]))
        position_check(res[0], res[2])

    numbers_to_generate = int(input(
        "how many epochs would you like to generate: "))

    generated = 0
    for i in range(numbers_to_generate):
        for i2 in range(len(cg.words)):
            ret = cg.generate(48, i2)
            img = ret[0]
            index = ret[1]
            pos = ret[2]
            index = index[0]
            filename = "{}{}_{}_{}.jpg".format(OUTPUT_DIR,
                                                  str(cg.words[index]),
                                                  index,
                                                  "{}-{}-{}-{}".format(
                                                      pos[0],
                                                      pos[1],
                                                      pos[2],
                                                      pos[3]
                                                  )
                                                  )
            cv2.imencode(".jpg", img)[1].tofile(filename)
            generated += 1
            print("\r{} image generated in \"{}\"".format(generated,
                                                          OUTPUT_DIR),
                  end="")



if __name__ == "__main__":
    init()
    main()
else:
    init()