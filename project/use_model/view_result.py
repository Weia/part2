# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 9:49
# @Author  : weic
# @FileName: view_result.py
# @Software: PyCharm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
with open('result.txt') as f:
    results=f.readlines()

    for result in results:
        #print(type(result))
        #print(result)
        list_result=result.split('*')
        #print(list_result)
        imgPath=list_result[0]
        #print(list_result[1].split(' '))
        label=list(map(float,list_result[1].split(' ')[:-1]))
        print(label)
        label=np.asarray(list((map(lambda x:x,label)))).reshape(-1,2)


        #show in range 39*39
        # print(label)
        # print(x)
        # print(y)
        # img=Image.open(imgPath).resize((39, 39),Image.ANTIALIAS)
        # plt.imshow(img)
        # plt.plot(x,y,'r*')
        # plt.show()

        #show in the orange image

        img=Image.open(imgPath)
        width,height=img.size
        x=label[:,0]*width/39.0
        y=label[:,1]*height/39.0

        plt.imshow(img)
        plt.plot(x,y,'r*')
        plt.show()

