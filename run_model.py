# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 21:31:19 2021

@author: asus
"""

import tensorflow as tf
import os

def run_unet(pd,mo):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    config = tf.ConfigProto()#对session进行参数配置
    config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./model/unet37000.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            xs=sess.graph.get_tensor_by_name("input_x:0")
            #xstate=sess.graph.get_tensor_by_name("input_state:0")
            mode=sess.graph.get_tensor_by_name("mode:0")
            
            state=sess.graph.get_tensor_by_name("output:0")
            #memory=sess.graph.get_tensor_by_name("new_memory:0")
            
            
            res=sess.run(state,feed_dict={xs:pd,mode:mo})   
            #o_memory=sess.run(memory,feed_dict={xs:pd,mode:mo})
    return res

