# -*- coding: utf-8 -*-
# @Time    : 2018/4/20 21:45
# @Author  : weic
# @FileName: train.py
# @Software: PyCharm
import tensorflow as tf
import os
import numpy as np
import models
import load_batch_data


import stage2Model
import stage3model

Points=5
ImageSize=39

model_dir='./model'#save model
train_log_dir='./log/train'#every batch info,loss and lr
epoch_log_dir='./log/epoch'#every epoch info,feature map

nEpoch=100000
batch_size=256
dataNum=13000 #num of samples
numBatch=5#dataNum//batch_size
save_stage=500 #step to save
learning_rate=0.00001
decay_steps=2000 #step to change lr
decay_rate=0.96# rate to change lr
change_lr=True #if change lr

data_file=r'/media/weic/新加卷/源码/face-point-dection-2013/trainImages/train.tfrecords' #dataset to train

def main():
    global_step=tf.Variable(0,trainable=False)
    print('*' * 10, 'create model', '*' * 10)
    with tf.name_scope('input'):
        inp = tf.placeholder(tf.float32, [None, ImageSize, ImageSize, 3], name='pl_input')
        label = tf.placeholder(tf.float32, [None, Points, 2], name='pl_label')
    models.local_share_weight_conv2(inp,2,[1,1,1,1],20,1,1)

    with tf.name_scope('model'):
        out_model = stage2Model.stage2_part(inp, 5)
    with tf.name_scope('loss'):
        diff = tf.subtract(out_model, label, name='sbu_label')
        l2_loss = tf.nn.l2_loss(diff, name='l2_loss')
        f_loss = tf.div(tf.cast(tf.sqrt(l2_loss * 2),tf.float32),ImageSize)
    with tf.name_scope('optimizer'):
        if change_lr:
            lr=tf.train.exponential_decay(learning_rate,global_step,decay_steps,decay_rate,staircase=True,name='learning_rate')#指数式衰减

        else:
            lr = learning_rate
        opt = tf.train.AdamOptimizer(lr)
        opti_min = opt.minimize(l2_loss,global_step=global_step)


    print('*' * 10, 'Done', '*' * 10)

    # 添加summary量
    with tf.name_scope('summary'):
        tf.summary.scalar('lr', lr, collections=['every_step'])
        tf.summary.scalar('loss', l2_loss, collections=['every_step'])
        train_summaries = tf.summary.merge_all('every_step')
        val_summarise=tf.summary.merge_all('epoch_step')
    #定义加载一个batch数据的tensor
    images,labels=load_batch_data.batch_samples(batch_size,data_file,shuffle=True)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #启动内存队列，加载图片到内存
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord,sess=sess)
        #查看是否有之前保存的模型，有加载
        ckpt=tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('have model ,load model ,continue train')
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            print('first run ,create model')
        #保存模型图结构

        train_file_writer=tf.summary.FileWriter(train_log_dir,graph=tf.get_default_graph())
        val_file_writer=tf.summary.FileWriter(epoch_log_dir)

        #IFSaveMate=True #if save meta info
        loss=[]
        for epoch in range(nEpoch):
            print('这是第%d个epoch'%(epoch+1))
            e_loss=0
            for batch in range(numBatch):
                print('这是第%d个batch'%(batch+1))
                #加载batch数据
                b_images,b_labels=sess.run([images,labels])
                # test_diff,test_label,test_output=sess.run([diff,label,out_model],feed_dict={inp: b_images, label: b_labels})
                # print('label',test_label)
                # print('out_model',test_output)
                # print('diff',test_diff)

                if batch%save_stage==0:

                    train_step=sess.run(global_step)
                    _,b_loss, str_summary,floss = sess.run([opti_min,l2_loss, train_summaries,f_loss],
                                                   feed_dict={inp: b_images, label: b_labels})
                    #写入训练的总结信息
                    train_file_writer.add_summary(str_summary,train_step)

                else:
                    _,b_loss,floss = sess.run([opti_min,l2_loss,f_loss],feed_dict={inp: b_images, label: b_labels})
                print('loss值为：%f'%(b_loss))
                loss.append(floss)

                e_loss+=b_loss
                break
            print('epoch loss is :%f'%(e_loss/numBatch))
            if epoch %save_stage==0:
                # 保存模型

                saver.save(sess, os.path.join(os.getcwd(), model_dir, 'model%d.ckpt' % ((epoch + 1) * epoch)))

            #写入一个epoch的汇总信息
            e_images,e_labels=sess.run([images,labels])
            str_epoch_summary=sess.run(val_summarise,feed_dict={inp:e_images,label:e_labels})
            val_file_writer.add_summary(str_epoch_summary,epoch)
        coord.request_stop()
        coord.join(threads)
    train_file_writer.flush()
    val_file_writer.flush()
    train_file_writer.close()
    val_file_writer.close()
    with open('./loss.txt', 'w') as f:
        for l in loss:
            f.write(str(l) + '\n')




if __name__ == '__main__':

    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

    if not (tf.gfile.Exists(train_log_dir) and tf.gfile.Exists(epoch_log_dir)):
        tf.gfile.MakeDirs(train_log_dir)
        tf.gfile.MakeDirs(epoch_log_dir)

    try:
        main()
    except Exception as info:
        print(info)
        exit()
