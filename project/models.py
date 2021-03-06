import tensorflow as tf
import numpy as np
import sys

#sys.setrecursionlimit(1000000)


"""
残差网络
"""

nMoudel=1#hourglass 中residual 模块的数量
LRNKernel=11

def batch_norm(input_images):
    # Batch Normalization批归一化
    # ((x-mean)/var)*gamma+beta
    #输入通道维数
    #parms_shape=[input_images.get_shape()[-1]]
    #parms_shape=tf.shape(input_images)[-1]
    ##print(parms_shape)
    #offset
    beta=tf.Variable(tf.constant(0.0,tf.float32),name='beta',dtype=tf.float32)
    #scale
    gamma=tf.Variable(tf.constant(1.0,tf.float32),name='gamma',dtype=tf.float32)
    #为每个通道计算均值标准差
    mean,variance=tf.nn.moments(input_images, [0, 1, 2], name='moments')
    y=tf.nn.batch_normalization(input_images,mean,variance,beta,gamma,0.001)
    y.set_shape(input_images.get_shape())

    return y


def batch_norm_relu(x):
    r_bn=batch_norm(x)
    r_bnr=tf.nn.relu(r_bn,name='relu')
    return  r_bnr


def conv2(input_images,filter_size,stride,out_filters,div_p=2,padding='SAME',weight=None,activate=tf.nn.relu):
    #将权重分模块做卷积，2 2分
    in_filters=input_images.get_shape().as_list()[-1]
    #卷积核初始化

    part_results=[]
    num_part=div_p*div_p
    shape_input=input_images.get_shape().as_list()
    part_even_w=shape_input[1]//2+filter_size-1
    part_even_h=shape_input[2]//2+filter_size-1
    part_odd_w=shape_input[1]//2
    part_odd_h=shape_input[2]//2
    region_list=[
        [0,part_even_w,0,part_even_h],
        [part_odd_w,shape_input[1],0,part_even_h],
        [0,part_even_w,part_odd_h,shape_input[2]],
        [part_odd_w,shape_input[1],part_odd_h,shape_input[2]]
    ]
    # print(region_list)
    for i in range(num_part):
        # print(i)
        biase = tf.Variable(tf.constant(0.0, shape=[out_filters]), dtype=tf.float32, name='biases'+str(i))
        if weight:
            _weights=weight
        else:
            _weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                            ([filter_size,filter_size,in_filters,out_filters])
                            ,name = 'weight1'+str(i))


        part_input = input_images[:, region_list[i][0]:region_list[i][1], region_list[i][2]:region_list[i][3], :]
        # print(part_input.get_shape().as_list())
        r_conv = tf.nn.conv2d(part_input, _weights, strides=stride, padding=padding)
        r_biases = tf.add(r_conv, biase)
        r_act = activate(r_biases)
        # print('result',r_act.get_shape().as_list())
        part_results.append(r_act)
    _01_result=tf.concat((part_results[0],part_results[1]),axis=1)
    _23_result=tf.concat((part_results[2],part_results[3]),axis=1)
    _result=tf.concat((_01_result,_23_result),axis=2)
    # print('01',_01_result.get_shape().as_list(),'\n',
    #       '23',_23_result.get_shape().as_list(),'\n',
    #       'result',_result.get_shape().as_list())
    relu_result=tf.nn.relu(_result,name='relu')
    return relu_result


def down_sampling(x,ksize=2,strides=2,padding='VALID'):

    #下采样
    #ksize: A 1-D int Tensor of 4 elements.
    #strides: A 1-D int Tensor of 4 elements
    return tf.nn.max_pool(x,[1,ksize,ksize,1],[1,strides,strides,1],padding=padding,name='max_pool')


def full_connect(inp,out_filters):
    #input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", name=None
    kernel_w=inp.get_shape().as_list()[1]
    kernel_h=inp.get_shape().as_list()[2]

    _weights = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                           ([kernel_w, kernel_h, inp.get_shape().as_list()[-1], out_filters])
                           , name='weight1')
    conv=tf.nn.conv2d(inp,_weights,[1,1,1,1],padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[out_filters], dtype=tf.float32), name='biases')
    re_conv2 = tf.add(conv, biases)
    #sum_conv=tf.reduce_sum(conv,[0,1,2])
    return re_conv2

def orid_conv2(inp,filter_size,out_filters,strides,padding='VALID'):
    #普通的卷积
    weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
        [filter_size,filter_size,inp.get_shape().as_list()[-1],out_filters]
    ),name='weights')
    biases=tf.Variable(tf.constant(0.0,shape=[out_filters],dtype=tf.float32),name='biases')
    conv=tf.nn.conv2d(inp,weights,[1,strides,strides,1],padding)
    re_conv2=tf.add(conv,biases)
    relu_result=tf.nn.relu(re_conv2,name='relu')
    return relu_result


def local_share_weight_conv2(input_images,filter_size,stride,out_filters,div_w=2,div_h=2,padding='VALID',weight=None,activate=tf.nn.relu):
    #将权重分模块做卷积
    in_filters=input_images.get_shape().as_list()[-1]
    #卷积核初始化

    part_results=[]
    num_part=div_w*div_h
    shape_input=input_images.get_shape().as_list()
    #每一部分的宽度和高度计算
    w_part_middle=shape_input[1]//div_w+filter_size-1
    h_part_middle=shape_input[2]//div_h+filter_size-1
    norm_part_width=shape_input[1]//div_w
    norm_part_height=shape_input[2]//div_h
    region_list=[]

    begin_w = 0
    begin_h = 0

    for part in range(num_part):
        change_line=False

        if (part+1)%div_w==0:
            part_width=shape_input[1]
            change_line=True
        else:
            part_width=w_part_middle+begin_w
        if num_part-(part+1)<div_w:
            part_height=shape_input[2]
        else:
            part_height=h_part_middle+begin_h
        region_list.append([begin_w,part_width,begin_h,part_height])

        begin_w = begin_w+norm_part_width if not change_line else 0
        begin_h = begin_h+norm_part_height if  change_line else begin_h
        print(begin_w,begin_h)

    for i in range(num_part):
        # print(i)
        biase = tf.Variable(tf.constant(0.0, shape=[out_filters]), dtype=tf.float32, name='biases'+str(i))
        if weight:
            _weights=weight
        else:
            _weights=tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)
                            ([filter_size,filter_size,in_filters,out_filters])
                            ,name = 'weight1'+str(i))

        part_input = input_images[:, region_list[i][0]:region_list[i][1], region_list[i][2]:region_list[i][3], :]
        r_conv = tf.nn.conv2d(part_input, _weights, strides=stride, padding=padding)
        r_biases = tf.add(r_conv, biase)
        r_act = activate(r_biases)
        part_results.append(r_act)
    hori_result=[]
    for r_i in range(len(part_results)):
        if r_i%div_w==0:
            hori_result.append(tf.concat(part_results[r_i:r_i+div_w],axis=1))

    _result=tf.concat(hori_result,axis=2)
    print(
          'result',_result.get_shape().as_list())
    relu_result=tf.nn.relu(_result,name='relu')
    return relu_result




