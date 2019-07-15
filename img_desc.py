# -*- coding: utf-8 -*-
"""
Created on Mon May  6 08:36:05 2019

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
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization,CuDNNLSTM
from keras import Input, layers
from keras import optimizers
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from my_class1 import DataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


sys.path.append('./darkflow-master')
sys.path.append('./word_encoding')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


#-----------------------------------------------------------------------------
#----------------INIT VAR----------------------------------------------------

desc_maxlen=50                                                
word_count_threshold = 10
word_counts = {}
nsents=0
ixtoword = {}
wordtoix = {}
batch_size=16
train_count=0

#-----------------------------------------------------------------------------
#----------------GLOVE----------------------------------------------------
embeddings_index = {}
filename = './word_encoding/glove.6B.200d.txt'
f = open(filename, encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Glove Word Vectors:', len(embeddings_index))

#-----------------------------------------------------------------------------
#----------------INCEPTION----------------------------------------------------
'''
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
model_temp = InceptionV3(weights='imagenet', include_top=True)
InceptionModel = Model(inputs=model_temp.input, outputs=model_temp.get_layer('avg_pool').output)
'''

#-----------------------------------------------------------------------------
#----------------LOAD DATA----------------------------------------------------
df=pd.read_csv('./Dataset 2/coco_descriptions.txt',sep='|')
cleaned_id=list()
cleaned_desc=list()
print("dataframe load:", df.shape)
for index,row in df.iterrows():
    print("LOAD",index)
    cleaned_id.append(row[0].strip())
    cleaned_desc.append('startseq '+row[1].strip()+' stopseq')

#print(cleaned_desc)
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


feature_vector=np.load('coco_feature_vector.npy')
print("feature_vector ", feature_vector.shape) 


#-------------------------------------------------------------------------------
            
embedding_dim = 200
# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

print("Embedding Matrix",embedding_matrix.shape)



inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(desc_maxlen,))
se1 = Embedding(vocab, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
inputs3 = Input(shape=(8,))
pe1 = Embedding(vocab, embedding_dim, mask_zero=True)(inputs3)
pe2 = Dropout(0.5)(pe1)
pe3 = LSTM(256)(pe2)
decoder1 = add([fe2, se3, pe3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

model.summary()
model.layers[3].set_weights([embedding_matrix])
model.layers[3].trainable = False
model.layers[4].set_weights([embedding_matrix])
model.layers[4].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='adam')
epochs = 10
batch_size = 16
'''
steps = len(cleaned_id)//number_pics_per_batch
for i in range(epochs):
    generator = data_generator(cleaned_desc, feature_vector, wordtoix, desc_maxlen, number_pics_per_batch)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
model.save('./model_weights/model.h5')
model.save_weights('./model_weights/modelwt.h5')
'''

training_generator = DataGeneratorYolo(encoder_input_data,feature_vector,cleaned_desc,ixtoword,wordtoix,desc_maxlen,vocab,batch_size)
#validation_generator = DataGeneratorYolo(val_feature_vector,val_cleaned_desc,ixtoword,wordtoix,desc_maxlen,vocab,batch_size)
model.fit_generator(generator=training_generator,
                    steps_per_epoch=ceil(train_count / batch_size),
                    use_multiprocessing=False,
                    workers=6, epochs=20)


model.save_weights('./model_weights/model_cap_wtonly.h5')
model.save('./model_weights/model_cap.h5')