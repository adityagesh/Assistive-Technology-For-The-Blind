# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:01:26 2019

@author: Aditya Nagesh
"""
import pandas as pd
import numpy as np

                                               
word_count_threshold = 10
word_counts = {}
nsents=0
ixtoword = {}
wordtoix = {}
batch_size=16
train_count=0
df=pd.read_csv('./Dataset 2/coco_descriptions.txt',sep='|')
cleaned_id=list()
cleaned_desc=list()
df=df.sample(frac=1).reset_index(drop=True)         #shuffle
print(df.shape)
for index,row in df.iterrows():
    print("LOAD",index)
    cleaned_id.append(row[0].strip())
    cleaned_desc.append(row[1].strip())
train_count=df.shape[0]
ix = 3

desc_maxlen=0

for sent in cleaned_desc:
    nsents+=1
    print(nsents)
    if (len(sent.split(" ")) > desc_maxlen):
        desc_maxlen=len(sent.split(" "))+1  #startseq and endseq
    #sent="startseq"+sent+" endseq"
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]


#word to index mapping
ix = 3
wordtoix['']=0
ixtoword[0]=''
wordtoix['startseq']=1
ixtoword[1]='startseq'
wordtoix['stopseq']=2
ixtoword[2]='stopseq'
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
print("vocab size ",len(ixtoword))

#dump using pickle
import pickle
with open('coco_wordtoix', 'wb') as fp:
    pickle.dump(wordtoix, fp)
with open('coco_ixtoword', 'wb') as fp:
    pickle.dump(ixtoword, fp)
print(ixtoword)