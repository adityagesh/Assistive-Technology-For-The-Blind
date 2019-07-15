# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:44:55 2019

@author: Aditya Nagesh
"""



import numpy as np
import pandas as pd
from math import ceil
from numpy import array
import sys
import time
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential,Model,load_model
from keras.layers import LSTM, Embedding, Dense,Activation, Flatten, Reshape, concatenate, Dropout,CuDNNLSTM
from keras import Input
from keras.layers.merge import add
from my_class1 import DataGeneratorYolo
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append('./darkflow-master')
sys.path.append('./word_encoding')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


desc_maxlen=50    

#--------------------------------------------------------------------------------
#----------------------YOLO INIT-----------------------------
from darkflow.net.build import TFNet
import cv2
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu":1}
tfnet = TFNet(options)


#--------------------------------------------------------------------------------
#----------------------LOAD DATA-----------------------------
df=pd.read_csv('./Dataset 2/coco_descriptions.txt',sep='|')
cleaned_id=list()
cleaned_desc=list()
print("dataframe load:", df.shape)
for index,row in df.iterrows():
    print("LOAD",index)
    cleaned_id.append(row[0].strip())
    cleaned_desc.append('startseq '+row[1].strip()+' stopseq')
train_count=len(cleaned_id)

#--------------------------------------------------------------------------------
#----------------------WORDTOIX and IXTOWORD MAPPING-----------------------------
import pickle
with open ('coco_wordtoix', 'rb') as fp:
    wordtoix = pickle.load(fp)
with open ('coco_ixtoword', 'rb') as fp:
    ixtoword = pickle.load(fp)

vocab=len(ixtoword)
print("vocab size ",vocab)
print("desc maxlen ",desc_maxlen)

yolo_out=list()
inception_out=list()
_status=0
for image_id in cleaned_id:
    _status+=1
    print("YOLO INCEPTION", _status ,train_count)
    path="D:/Dataset 2/val2014/"+str(image_id)   #relative path giving error
    imgcv = cv2.imread(path)
    result = tfnet.return_predict(imgcv)
    sublist=list()
    obj_count=0
    for object in result:
        temp1=list()
        label=object['label']
        if label in wordtoix.keys():
            if(obj_count>7):
                break;
            obj_count+=1
            temp1.append(np.array([wordtoix[label]]))
            sublist.append(temp1)
    while(obj_count<=7):
        obj_count+=1
        sublist.append(np.array(np.zeros((1),dtype='float32')))
    temp_input=np.asarray(sublist).reshape((8,))
    yolo_out.append(np.array(temp_input)) 
encoder_input_data=np.asarray(yolo_out) 
np.save('coco_encoder_input_labelonly',encoder_input_data)