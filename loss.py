# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:02:26 2021

@author: asus
"""

import tensorflow as tf
import numpy as np

def get_loss(gt_target,gt_label,net_score):
    net_score0=tf.reshape(net_score,[-1])
    
    gt_label=tf.reshape(gt_label,[-1])
    gt_target=tf.reshape(gt_target,[-1])
    
    t_select=tf.where(tf.not_equal(gt_target,-1))
    net_score0=tf.reshape(tf.gather(net_score0,t_select),[-1])
    
    gt_label=tf.reshape(tf.gather(gt_label,t_select),[-1])
    
    cls_sq0=tf.reduce_mean(tf.square(net_score0-gt_label))
    
    cls_loss=cls_sq0
    
    loss=cls_loss
    
    return loss