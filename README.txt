1. Download COCO Dataset from http://cocodataset.org/#download
		
2014 Train images [83K/13GB]
2014 Val images [41K/6GB]
2014 Test images [41K/6GB]
2014 Train/Val annotations [241MB]
2014 Testing Image info [1MB]

Extract to folder 'Dataset 2'


1. Dataset Process.py
   Use COCO API to fetch image file name and description.
	Clean the descriptions.
	Create descriptions.txt with the format <filename.jpg | descripiton> for each image in train set.
	
2. Vocab.py
	Load the data from descriptions.txt
	Create the vocabulary using words that occur more than 10 occurance
	Integer encode each word in vocabulary
	Find the maximum length of description
	Save word to index mapping to 'coco_wordtoix' using pickle dump
	Save index to word mapping to 'coco_ixtoword' using pickle dump

3.Inception.py
	Load the data from descriptions.txt
	For each image, generate feature vector using pretrained Inception v3.
	Save feature vector to hard disk using np.save to file 'coco_feature_vector.npy'

4.yolo.py
	Load filenames from descriptions.txt
	For each image, create LSTM input with upto 8 objects
	Save the array to hard disk using np.save to file 'coco_encoder_input_labelonly.npy'
5.GeneratorClass.py
	Create class to create data set for fit_generator


6.img_desc.py
	Load descriptions, wordtoix, ixtoword, feature_vector.
	Create embedding matrix for each word in vocabulary using pretrained Glove Model (200D)
	Define the model and compile it
	Train the model using fit_generator
	
7.predict.py
	Load filenames of test dataset
	Load wordtoix and ixtoword
	Create feature vector for each image
	Pass feature vector and partial description to the model 
	Pass the output to pyttsx3
	
	