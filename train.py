# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:11:38 2021

@author: asus
"""

import numpy as np
import tensorflow as tf
from unet import Unet
from flags import flags
from loss import get_loss
from support.read_data import get_train_data
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


lr=0.01

input_x=tf.placeholder(tf.float32,[None,640,960,3],name="input_x")
input_x=input_x/255.0

label=tf.placeholder(tf.float32,[None,640,960],name="label")
ignore=tf.placeholder(tf.float32,[None,640,960],name="ignore")

mode=tf.placeholder(tf.float32,name="mode")
mode=tf.cast(mode,tf.bool)

unet=Unet(input_x)
pred=unet.make_unet(mode,flags)
output=tf.reshape(pred,[-1,640,960],name="output")

loss=get_loss(ignore,label,pred)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.9)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session(config=config) as sess:

    sess.run(init)
    
    for i in range(50001):
        sjsi=np.random.randint(0,420,1)[0]
        ocs,no,pic=get_train_data(sjsi)
        sess.run(train_step,feed_dict={input_x:pic,label:ocs,ignore:no,mode:1})
        l=sess.run(loss,feed_dict={input_x:pic,label:ocs,ignore:no,mode:1})
        print(i,l)
        if i%1000==0:
            output_graph_def=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['output'])#sess.graph_def
            with tf.gfile.FastGFile('./model/unet'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def.SerializeToString())