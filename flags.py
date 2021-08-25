# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:22:29 2021

@author: asus
"""

class Flags():
    def __init__(self,epochs,batch_size,reg):
        self.epochs = epochs
        self.batch_size=batch_size
        self.reg=reg
        
flags=Flags(5001,1,0.1)