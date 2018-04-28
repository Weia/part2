# -*- coding: utf-8 -*-
# @Time    : 2018/4/19 20:55
# @Author  : weic
# @FileName: stage2Model.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import models
nPoints=5





def stage2_part(inp,num_point):
    """
    stage2 predict,use small part as input to implement more precise location
    :param inp: a tensor with 4-D 39*39
    :param num_point: a int number, the count of point to locate
    :return:a tensor 1-D,[num_point*2]
    """
    conv_1 = models.conv2(inp, 4, [1, 1, 1, 1], 20, padding='VALID')

    tf.summary.image('conv1',tf.transpose(conv_1[0:1,:,:,:],[3,1,2,0]),max_outputs=20,collections=['epoch_step'])
    pool_1 = models.down_sampling(conv_1)

    conv_2 = models.conv2(pool_1, 3, [1, 1, 1, 1], 40, padding='VALID')

    tf.summary.image('conv2',tf.transpose(conv_2[0:1,:,:,:],[3,1,2,0]),max_outputs=40,collections=['epoch_step'])
    pool_2 = models.down_sampling(conv_2)

    conv_3 = models.conv2(pool_2, 3, [1, 1, 1, 1],  60, padding='VALID')

    tf.summary.image('conv3',tf.transpose(conv_3[0:1,:,:,:],[3,1,2,0]),max_outputs=60,collections=['epoch_step'])
    pool_3 = models.down_sampling(conv_3)

    conv_4 = models.conv2(pool_3, 2, [1, 1, 1, 1],  80, padding='VALID')

    tf.summary.image('conv4',tf.transpose(conv_4[0:1,:,:,:],[3,1,2,0]),max_outputs=80,collections=['epoch_step'])


    fc_1=models.full_connect(conv_4,120)
    #print('full_connect1',fc_1.get_shape().as_list())
    fc_2=models.full_connect(fc_1,num_point*2)
    #print('full_connect2', fc_2.get_shape().as_list())
    sum_conv=tf.reduce_sum(fc_2,[1,2])
    #print('full_connect2', sum_conv.get_shape().as_list())

    pre_label=tf.cast(tf.reshape(sum_conv,[-1,nPoints,2],'output'),tf.float32)
    #print('full_connect2', pre_label.get_shape().as_list())


    return pre_label


