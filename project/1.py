# -*- coding: utf-8 -*-
# @Time    : 2018/4/20 17:01
# @Author  : weic
# @FileName: 1.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import models
if __name__ == '__main__':
    #feed_input = np.random.uniform(0.0, 1.0, [3,38, 38,  20])
    # feed_input=np.ones([2,2])
    # b=np.eye(2,2)
    # c=tf.subtract(feed_input,b)
    # ls_loss=tf.nn.l2_loss(c)
    # # input = tf.placeholder(tf.float32, [3,38, 38,  20], name='t_fc_input')
    # # a=models.conv2(input,3,[1,1,1,1],40,div_p=2,padding='VALID')
    #
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     print(sess.run(ls_loss))

    a=np.asarray([[1,23,3],
                  [1, 23, 3]])
    print(a)
    print(a[0:4])