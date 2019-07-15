# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:09:30 2019

@author: Aditya Nagesh
"""
import pandas as pd
import numppy as np

_status=0
inception_out=list()
train_count=0
df=pd.read_csv('./Dataset 2/coco_descriptions.txt',sep='|')
cleaned_id=list()
cleaned_desc=list()
print("dataframe load:", df.shape)
for index,row in df.iterrows():
    print("LOAD",index)
    cleaned_id.append(row[0].strip())
    cleaned_desc.append(row[1].strip())
train_count=len(cleaned_id)


#inceptionv3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
model_temp = InceptionV3(weights='imagenet', include_top=True)
InceptionModel = Model(inputs=model_temp.input, outputs=model_temp.get_layer('avg_pool').output)
#model.summary()


for image_id in cleaned_id:
    _status+=1
    print("YOLO INCEPTION", _status ,train_count)
    path="D:/Dataset 2/train2014/"+str(image_id)   #relative path giving error
    img = image.load_img(path, target_size=(299, 299))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    inception_feature=InceptionModel.predict(img_data)
    inception_out.append(inception_feature)
feature_vector=np.array(inception_out)
feature_vector=np.reshape(feature_vector,(train_count,2048))
np.save('coco_feature_vector',feature_vector)