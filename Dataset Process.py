# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:14:29 2019

@author: Aditya Nagesh
"""

import os
import string
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

table= str.maketrans('', '', string.punctuation)  #remove punctuation

# initialize COCO API for instance annotations
dataDir = '.'
dataType = 'train2014'
instances_annFile = os.path.join(dataDir, 'cocoapi/annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'cocoapi/annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())
print(len(ids))
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
filename_list=list()
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
lines=list()
for i in ids:
    img_id=coco.anns[i]['image_id']
    img = coco.loadImgs(img_id)[0]
    filename=img['file_name']
    if(filename not in filename_list):                      #TO AVOID REPEAT 
        annIds = coco_caps.getAnnIds(imgIds=img['id']);
        anns = coco_caps.loadAnns(annIds)
        for j in anns:
            desc=j['caption']
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc =  ' '.join(desc)
            lines.append(filename +' | '+desc)
            print(filename,desc)
        filename_list.append(filename)
    #coco_caps.showAnns(anns)   
data='\n'.join(lines)
file=open('./Dataset/coco_val_descriptions.txt','w')
file.write(data)
file.close()
