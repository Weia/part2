import tensorflow as tf
"""加载一个batchsize的image"""
WIDTH=39
HEIGHT=39
HM_HEIGHT=64
HM_WIDTH=64
Points=5
def _read_single_sample(samples_dir):
    filename_quene=tf.train.string_input_producer([samples_dir])
    reader=tf.TFRecordReader()
    _,serialize_example=reader.read(filename_quene)
    features=tf.parse_single_example(
        serialize_example,
        features={
                    'label':tf.FixedLenFeature([Points*2],tf.float32),
                    'image':tf.FixedLenFeature([],tf.string)
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [HEIGHT,WIDTH, 3])#！reshape 先列后行
    label = tf.cast(features['label'], tf.float32)
    return image,label
    # print(img.shape)
    # print(label)


def batch_samples(batch_size,filename,shuffle=False):
    """
    filename:tfrecord文件名
    """

    image,label=_read_single_sample(filename)
    label=tf.reshape(label,[-1,2])
    #label = gene_hm.resize_label(label)#将label放缩到64*64
    #label=gene_hm.tf_generate_hm(HM_HEIGHT, HM_WIDTH ,label, 64)
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, min_after_dequeue=1000,num_threads=2,capacity=28000)
    else:
        image_batch,label_batch=tf.train.batch([image,label],batch_size, num_threads=2)

    return image_batch,label_batch



# # # """测试加载图像"""
import matplotlib.pyplot as plt
#import load_batch_data
from PIL import Image
#import numpy as np
#from pyheatmap import HeatMap
#from  pyheatmap.heatmap import HeatMap
#import HeatMap

# with tf.Session() as sess: #开始一个会话
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     # image,label=read_single_sample('test_code.tfrecords')
#     image_batch,label_batch=batch_samples(15,r'/media/weic/新加卷/源码/face-point-dection-2013/trainImages/train.tfrecords',True)
#     #image_batch,label_batch=tf.train.batch([image,label], batch_size=3, capacity=200, num_threads=2)
#
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#
#     example, l = sess.run([image_batch, label_batch])
#
#     for i in range(15):
#           # 在会话中取出image和label
#
#         img=Image.fromarray(example[i], 'RGB')#这里Image是之前提到的
#         print(l[i])
#         x=l[i].reshape(-1,2)[:,0]
#         y=l[i].reshape(-1,2)[:,1]
#         plt.imshow(img)
#         plt.plot(x,y,'r*')
#
#         plt.show()
# #
# coord.request_stop()
# coord.join(threads)
