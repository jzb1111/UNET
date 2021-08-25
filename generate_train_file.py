# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 23:00:39 2021

@author: asus
"""

from support.read_data import generate_train_file,file_name,get_batch_by_num
import numpy as np

r,d,f=file_name('./VOC2007/SegmentationClass/')

for i in range(len(f[0])):
    print(i)
    pic,gld=get_batch_by_num(i)
    gld=gld[0]
    ocs,no=generate_train_file(gld)
    np.save('./train_file/ocs_'+str(i),ocs)
    np.save('./train_file/no_'+str(i),no)
    np.save('./train_file/pic_'+str(i),pic)
print('ok')