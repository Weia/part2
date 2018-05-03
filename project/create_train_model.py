# -*- coding: utf-8 -*-
# @Time    : 2018/5/3 8:52
# @Author  : weic
# @FileName: create_train_model.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import stage2Model,stage3model

def create_model(inp,nPoints):
    #第二阶段
    w_inp=inp.get_shape().as_list()[1]
    h_inp=inp.get_shape().as_list()[2]

    list_final_output=[]
    stage2output=stage2Model.stage2_part(inp,nPoints)#[-1,nPoints,2]
    num_samples=stage2output.get_shape().as_list()[0]
    for i in range(num_samples):
        #切出较小的区域，作为第三阶段输入
        print(i)
        list_sample_result=[]
        for j in range(nPoints):
            x=tf.cast(stage2output[i][j][0],tf.int32)
            y=tf.cast(stage2output[i][j][1],tf.int32)
            pad_inp=tf.pad(inp,[[0,0],[8,8],[8,8],[0,0]])
            stage3input=tf.reshape(pad_inp[i,x-7:x+8,y-7:y+8,:],[-1,15,15,3])
            if stage3input is None:
                stage3input=np.zeros([1,15,15,3])
                #stage3input=tf.slice(inp,)
            #stage3input=tf.placeholder(tf.float32,[1,15,15,3])
            stage3output=stage3model.stage3_part(stage3input)#[-1,1,2]
            list_sample_result.append(stage3output)
        sample_result=tf.concat(list_sample_result, axis=1)
        list_final_output.append(sample_result)
    final_output=tf.concat(list_final_output,axis=0)
    return final_output



    pass
