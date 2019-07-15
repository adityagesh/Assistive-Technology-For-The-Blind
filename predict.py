# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:22:59 2019

@author: Aditya Nagesh
"""

import pyttsx3
import numpy as np
import os
import time
from random import shuffle
#inceptionv3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import Model,load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#------------------------------INCEPTION v3------------------------------------
model_temp = InceptionV3(weights='imagenet', include_top=True)
InceptionModel = Model(inputs=model_temp.input, outputs=model_temp.get_layer('avg_pool').output)

#---------------------------------YOLO----------------------------------------
#yolo
from darkflow.net.build import TFNet
import cv2
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu":1}
tfnet = TFNet(options)


#--------------------------------LOAD TRAINED MODEL----------------------------
#without yolo './model_weights/model_cap.h5'
#with yolo  './model_weights/model_cap_yolo.h5'
model_yolo=load_model('./model_weights/model_cap_yolo.h5')
model_yolo.summary()

#----------------------------------------------------------------------------
#-------------------------------PYTTSX3--------------------------------------

engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed percent (can go over 100)
engine.setProperty('volume', 0.9)  # Volume 0-1

#-----------------------------------------------------------------------------
#--------------------------------SEARCH---------------------------------------


def SearchYolo(photo,yolo):
    in_text = 'startseq'
    for i in range(desc_maxlen):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=desc_maxlen)
        yhat = model_yolo.predict([photo,sequence,yolo], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'stopseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



#-----------------------------------------------------------------------------
#----------------------------------LOAD TEST IMAGE----------------------------
    
path='D:/Dataset 2/test2014/'
images=os.listdir(path)
shuffle(images)


#-----------------------------------------------------------------------------
#-------------------------WORDTOIX AND IXTOWORD-------------------------------

desc_maxlen=50
import pickle
with open ('coco_wordtoix', 'rb') as fp:
    wordtoix = pickle.load(fp)
with open ('coco_ixtoword', 'rb') as fp:
    ixtoword = pickle.load(fp)

vocab=len(ixtoword)
print("vocab size ",vocab)
print("desc maxlen ",desc_maxlen)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
for img in images:
    path='D:/Dataset 2/test2014/'+ img
    print(path)
#--------------------------------------------------------    
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()
#--------------------------------------------------------
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
    yolo_out=np.asarray(sublist).reshape((1,8))
#--------------------------------------------------------
    img = image.load_img(path, target_size=(299, 299))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    inception_feature=InceptionModel.predict(img_data)
#--------------------------------------------------------
    #caption=Search(inception_feature)
    #print(caption)
    caption=SearchYolo(inception_feature,yolo_out)
    print(caption)
    #caption=SearchYoloAll(inception_feature,yolo_out)
    #print(caption)
    engine.say(caption)
    engine.runAndWait()
    time.sleep(4)
