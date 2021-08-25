# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 20:57:03 2021

@author: asus
"""

import numpy as np
import os
import cv2

def file_name(file_dir):
    root_=[]
    dirs_=[]
    files_=[]
    for root, dirs, files in os.walk(file_dir):  
        #print(root) #当前目录路径  
        #print(dirs) #当前路径下所有子目录  
        #print(files) #当前路径下所有非目录子文件
        root_.append(root)
        dirs_.append(dirs)
        files_.append(files)
    return root_,dirs_,files_

def load_image(imageurl):#加载vgg模型必须这样加载图像
    im=cv2.resize(cv2.imdecode(np.fromfile(imageurl,dtype=np.uint8),cv2.IMREAD_COLOR),(960,640)).astype(np.float32)

    return im

def get_batch_by_num(num,is_train='train'):
    if is_train=='train':
        path1='./VOC2007/JPEGImages/'
        path2='./VOC2007/SegmentationClass/'
    else:
        path1='./VOC2007/JPEGImages/'
        path2='./VOC2007/SegmentationClass/'
    r1,d1,f1=file_name(path1)
    r2,d2,f2=file_name(path2)
    #fpic=f1[0]
    #print(f2)
    floc=f2[0]
    #print(floc)
    #sjs=np.random.randint(0,len(floc),size)
    sjs=num
    pic_data=[]
    loc_data=[]
    #for i in range(num):
    #loc_data.append(cv2.imdecode(np.fromfile(path2+floc[sjs],dtype=np.uint8),cv2.IMREAD_COLOR).astype(np.float32))
    loc_data.append(load_image(path2+floc[sjs]))
    picname=path1+floc[sjs].replace('png','jpg')
        #print(picname)
    pic_data.append(load_image(picname))
    #print(pic_data)
    return pic_data,loc_data

def generate_train_file(gld):
    
    gld=gld[:,:,0]+gld[:,:,1]+gld[:,:,2]
    
    out_sample=np.zeros((1,640,960))
    out_ignore=np.full((1,640,960),-1)
    #for i in range(len(gld)):
    right_num=0
    neg_count=0
    for h in range(640):
        for w in range(960):
            if gld[h][w]>0:
                out_sample[0][h][w]=1
                right_num+=1
                out_ignore[0][h][w]=0
            if gld[h][w]==0:
                neg_count+=1
    
        
    neg_num=0
    count_time=0
    if neg_count<=right_num:
        while count_time<right_num*3:
            sjsh=np.random.randint(0,640)
            sjsw=np.random.randint(0,960)
            if gld[sjsh][sjsw]==0:
                out_sample[0][sjsh][sjsw]=0
                neg_num+=1
                out_ignore[0][sjsh][sjsw]=0
            count_time+=1
    else:
        while neg_num<=right_num:
            sjsh=np.random.randint(0,640)
            sjsw=np.random.randint(0,960)
            if gld[sjsh][sjsw]==0:
                out_sample[0][sjsh][sjsw]=0
                neg_num+=1
                out_ignore[0][sjsh][sjsw]=0
    #gt_rect=runget4point(np.reshape(gld,[1,224,224]).astype(np.float32))
    return out_sample,out_ignore

def get_train_data(num):
    ocs=np.load('./train_file/ocs_'+str(num)+'.npy')
    no=np.load('./train_file/no_'+str(num)+'.npy')
    pic=np.load('./train_file/pic_'+str(num)+'.npy')
    return ocs,no,pic