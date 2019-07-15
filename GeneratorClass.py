# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:18:56 2019

@author: Aditya Nagesh
"""
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,feature_vector,cleaned_desc,ixtoword,wordtoix,desc_maxlen, vocab_size, batch_size=16, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.wordtoix=wordtoix
        self.ixtoword=ixtoword
        self.desc_maxlen = desc_maxlen
        self.cleaned_desc=cleaned_desc
        self.vocab_size = vocab_size
        self.feature_vector = feature_vector
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.cleaned_desc) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        cleaned_desc_data=[self.cleaned_desc[k] for k in indexes]
        feature_vector_data=[self.feature_vector[k] for k in indexes]
        # Find list of IDs
        #list_IDs_temp = [k for k in indexes]

        # Generate data
        X1, X2, y = list(), list(), list()
        X1, X2,y = self.__data_generation(cleaned_desc_data,feature_vector_data)

        return [array(X1), array(X2)],array(y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.cleaned_desc))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,cleaned_desc_data,photos):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1, X2, y = list(), list(), list()
        n=0
        # loop for ever over images
        for sentcount,desc in enumerate(cleaned_desc_data):
            photo=photos[sentcount]
            n+=1
            # encode the sequence
            seq = [self.wordtoix[word] for word in desc.split(' ') if word in self.wordtoix.keys()]
            # split one sequence into multiple X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.desc_maxlen)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                # store
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
        return array(X1), array(X2), array(y)
    
    
class DataGeneratorYolo(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,encoder_input_data,feature_vector,cleaned_desc,ixtoword,wordtoix,desc_maxlen, vocab_size, batch_size=16, shuffle=True):
        'Initialization'
        self.encoder_input=encoder_input_data
        self.batch_size = batch_size
        self.wordtoix=wordtoix
        self.ixtoword=ixtoword
        self.desc_maxlen = desc_maxlen
        self.cleaned_desc=cleaned_desc
        self.vocab_size = vocab_size
        self.feature_vector = feature_vector
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.cleaned_desc) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        cleaned_desc_data=[self.cleaned_desc[k] for k in indexes]
        feature_vector_data=[self.feature_vector[k] for k in indexes]
        encoder_input_data=[self.encoder_input[k] for k in indexes]
        # Find list of IDs
        #list_IDs_temp = [k for k in indexes]

        # Generate data
        X1, X2, X3, y = list(), list(), list(), list()
        X1, X2, X3, y = self.__data_generation(cleaned_desc_data,feature_vector_data,encoder_input_data)

        return [array(X1), array(X2), array(X3)],array(y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.cleaned_desc))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,cleaned_desc_data,photos,encoder_inputs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1, X2, X3, y = list(), list(), list(), list()
        n=0
        # loop for ever over images
        for sentcount,desc in enumerate(cleaned_desc_data):
            photo=photos[sentcount]
            encoder_input=encoder_inputs[sentcount]
            n+=1
            # encode the sequence
            seq = [self.wordtoix[word] for word in desc.split(' ') if word in self.wordtoix.keys()]
            # split one sequence into multiple X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.desc_maxlen)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                # store
                X1.append(photo)
                X2.append(in_seq)
                X3.append(encoder_input)
                y.append(out_seq)
        return array(X1), array(X2), array(X3), array(y)


