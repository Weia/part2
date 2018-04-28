import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
#加载模型，进行预测
result=open('./result.txt','w+')
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(r'../model/')  # 通过检查文件锁定最新模型,时间
    if ckpt and ckpt.model_checkpoint_path:#ckpt.model_checkpoint_path最新的模型

        new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')  # 载入图结构
        new_saver.restore(sess,ckpt.model_checkpoint_path)
        # for val in tf.trainable_variables():
        #     print(val.name, val.value)
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name('input/pl_input:0')
        # loss=graph.get_tensor_by_name('loss/cross_entropy_loss:0')
        output = graph.get_tensor_by_name('model/output:0')
        path=r'/media/weic/新加卷/源码/face-point-dection-2013/trainImages/lfw_5590'
        imgNames=os.listdir(path)

        for name in imgNames:
            imgPath=os.path.join(path,name)
            image=Image.open(imgPath)

            image=image.resize((39,39),Image.ANTIALIAS)

            images=np.expand_dims(image,0)
            print(images.shape)

        # # label=0


        # writer = tf.summary.FileWriter('load', graph=graph)
        # writer.flush()
        # #打印所有变量
        # # for op in graph.get_operations():
        # #     print(op.name,' ')

            test=sess.run(output,feed_dict={input:images})
            result.write(imgPath+'*')
            print(test)

            for i in range(5):

                result.write(str(test[0][i][0])+' ')
                result.write(str(test[0][i][1])+' ')
            result.write('\n')


    #loss=sess.run(graph.get_tensor_by_name('loss/train_loss:0'),feed_dict={'input_image':image,'labels':label})
    # print(loss)
