# -*- coding: utf-8 -*-
# @Time    : 2018/4/28 13:09
# @Author  : weic
# @FileName: stage3model.py
# @Software: PyCharm
import tensorflow as tf
import models


def stage3_part(inp):
    conv1=models.orid_conv2(inp,4,20,1)
    pool1=models.down_sampling(conv1)

    conv2=models.orid_conv2(pool1,3,40,1)
    pool2=models.down_sampling(conv2)

    fc1=models.full_connect(pool2,60)
    fc2=models.full_connect(fc1,2)

    re_conv=tf.reduce_sum(fc2,[1,2])
    pre_labels=tf.cast(tf.reshape(re_conv,([-1,1,2])),tf.float32)
    return pre_labels
