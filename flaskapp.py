# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:40:20 2019

@author: Aditya Nagesh
"""

path='D:/static/image.jpg'

import sys
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
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
import tensorflow as tf
from keras import backend as K



app=Flask(__name__)
CORS(app)

from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
sess = tf.Session(config=config)  
set_session(sess)  # set this TensorFlow session as the default session for Keras.

#------------------------------INCEPTION v3------------------------------------
model_temp = InceptionV3(weights='imagenet', include_top=True)
InceptionModel = Model(inputs=model_temp.input, outputs=model_temp.get_layer('avg_pool').output)
global graph1
graph1 = tf.get_default_graph()
#---------------------------------YOLO----------------------------------------
#yolo
from darkflow.net.build import TFNet
import cv2
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu":1}
tfnet = TFNet(options)
#--------------------------------LOAD TRAINED MODEL----------------------------

model_yolo=load_model('./model_weights/model_cap_yolo.h5')
global graph2
graph2 = tf.get_default_graph()

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
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

@app.route('/capture',methods=['POST'])
def recieve_image():
    message=request.get_json(force=True)
    capture_image=bytes(message, encoding="ascii")
    im = Image.open(BytesIO(base64.b64decode(capture_image)))
    im.save('./static/image.jpg')
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
                break
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
    with graph1.as_default():
        inception_feature=InceptionModel.predict(img_data)
#--------------------------------------------------------
    with graph2.as_default():
        caption=SearchYolo(inception_feature,yolo_out)
    response={
        'caption':caption
    }
    return jsonify(response)
app.run(debug=True,host='0.0.0.0',port=8085)