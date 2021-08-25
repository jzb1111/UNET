# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 21:42:53 2021

@author: asus
"""

from run_model import run_unet
import matplotlib.pyplot as plt
from support.read_data import get_train_data
import numpy as np

#mnist=gen_mnist_data()

ocs,no,pic=get_train_data(401)
#pic_re=np.reshape(pic,[28,28])
#init_state=np.ones([1,64])
#init_memory=np.ones([1,64])
#state=init_state
#memory=init_memory

res=run_unet(pic/255.0,1)
plt.figure(0)
plt.imshow(pic[0]/255)
plt.figure(1)
plt.imshow(res[0])
#print(res)    
#print(list(res[0]).index(max(res[0])))