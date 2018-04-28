# -*- coding: utf-8 -*-
# @Time    : 2018/3/19 10:19
# @Author  : weic
# @FileName: img2tfrecords.py
# @Software: PyCharm
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from process import process_img
import csv
WIDTH=39
HEIGHT=39

def _read_images(img_list):
    """读取csv文件的images信息"""
    print(2)
    f=open(img_list,'r')
    data=csv.reader(f)
    # with open(img_list,'r') as f:
    #     data = csv.reader(f)
    # f.close()

    return data
def _parse_label(line):
    #输入['241_135_1', '301_135_1']
    #将-1_-1_-1类型的label解析为-1 -1 -1 并返回数值类型的list

    #取x,y坐标
    p_label = np.asarray([float(x) for x in line]).reshape(-1,2)


    return p_label
def _parse_image(image):
    """解析一行images信息"""

    imgName = image[0]#图像名

    label = image[5:]

    parse_label=_parse_label(label)

    return imgName,parse_label

def img2TfRecord(img,label):
    """
    将image和label转换成tfrecord的example
    :param img:
    :param label: float list
    :return:
    """
    # 转换成二进制
    f_label = (label.reshape(1, -1)).tolist()[0]

    img_raw = img.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=f_label)),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
    s_example=example.SerializeToString()
    return s_example

def createTfRecords(img_dir,filename,img_list):
    """
        根据保存图像名和label的txt文件生成tfrecords文件

        :param img_dir: 存储图像路径
        :param filename: 要生成tfrecords文件名
        :param img_list: 保存图像的list
    """
    writer=tf.python_io.TFRecordWriter(filename)
    lines=_read_images(img_list)
    #imgNum=len(images)
    i=1

    for line in lines:
        print(i)
        #解析image获得图像名和label
        image=line[0].split(' ')

        imgName,label=_parse_image(image)


        print(label.shape)
        or_image=Image.open(os.path.join(img_dir,imgName))
        #从图像中裁剪出人
        #img,label=process_img.crop_data(or_image,label)
        #print(real_image)
        #将图像转换成256*256
        re_image,re_label=process_img.resize_image(or_image,label,WIDTH,HEIGHT)

        #将image和label转换成example
        example=img2TfRecord(re_image,re_label)
        writer.write(example)
        i+=1


    writer.close()


createTfRecords(r'E:\源码\face-point-dection-2013\trainImages',
                r'E:\源码\face-point-dection-2013\trainImages\train.tfrecords',
                r'E:\源码\face-point-dection-2013\trainImages\trainImageList.txt')