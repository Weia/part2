# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 9:49
# @Author  : weic
# @FileName: view_result.py
# @Software: PyCharm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
with open(r'E:\源码\face-point-dection-2013\trainImages\trainImageList.txt') as f:
    results=f.readlines()

    for result in results:
        #print(type(result))
        #print(result)
        list_result=result.split(' ')
        #print(list_result)
        #处理linux下的result文件

        imgName=list_result[0]
        imgPath=os.path.join((r'E:\源码\face-point-dection-2013\trainImages'),str(imgName))

        #处理windows下的result文件

        #imgPath=list_result[0]

        label=[float(x) for x in list_result[5:]]

        #在256*256的图上展示
        """
        label=np.asarray(list((map(lambda x:x*256/64,label)))).reshape(-1,2)
        x=label[:,0]
        y=label[:,1]
        img=Image.open(imgPath).resize((256, 256),Image.ANTIALIAS)
        plt.imshow(img)
        plt.plot(x,y,'r*')
        plt.show()"""

        #在原图上展示结果
        label=np.asarray(label).reshape(-1,2)
        x=label[:,0]
        y=label[:,1]
        img=Image.open(imgPath)
        width,height=img.size

        plt.imshow(img)
        plt.plot(x, y, 'r*')
        plt.show()



