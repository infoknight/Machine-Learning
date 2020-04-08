#!/usr/bin/env python3
#Deep Learning : Convolutional Neural Network

#--------------------------------------Data Preprocessing---------------------------------------------------------------------------#
#This step is not applicable to Image classification. However, the image pre-processing could be done by the following steps:-
#In this example we have thousands of images of dogs and cats. 
#1. Segregate the images of dogs and cats.
#2. Name the images as dog<num>.jpg and cat<num>.jpg
#3. Segregate the images and store them as Training_set and Test_set.

#--------------------------------------Convolutional Neural Netwok------------------------------------------------------------------#
#Importing the keras Libraries & Packages
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Convolution2D     #Deprecated in Keras 2 API. In stead use, Conv2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Initializing the CNN
classifier = Sequential()

#Step - 1: Convolution
#classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = "relu"))   : Deprecated in Keras 2 
            #32, 3, 3 ==> 32 Feature Detectors of 3 rows & 3 columns; make it 64 / 128 / 256 in GPUs
            #(64, 64, 3) ==> Standardize the images with 64 x 64 pixels and 3 colours  : For Tensorflow backend
            #(3, 64, 64) ==>                                                           : For Theano backend
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

#Step -2: Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
            #Pooling could be done in many ways : max, avg, etc

#Adding Second Convolution Layer        :This approach is much faster than adding second Hidden layer
#Second Convolution Layer must be added after the Pooling Layer and also followed by a Pooling Layer
classifier.add(Conv2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step -3: Flattening
classifier.add(Flatten())

#Step -4: Full-Connection
classifier.add(Dense(units = 128, activation = "relu"))         #Hidden Layer with 128 Neurons
classifier.add(Dense(units = 1, activation = "sigmoid"))        #Output Layer

#Compiling the CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
            #adam = Stochastic Gradient Descent Optimizer

#Fitting the CNN to the Images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(         #Image Augmentation: Eliminates the need for large image set by transorming the limited
        rescale=1./255,                     #image set with various mathematical functions
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../Data/22_cnn_dataset/training_set',
        target_size=(64, 64),               #Should be same as the input_shape supplied in Conv2D function; GPU => 256, 256
        batch_size=32,                      #GPU => 128 / 256 ...
        class_mode='binary')                #binary ==> since our ouput is either "dogs" or "cats"

test_set = test_datagen.flow_from_directory(
        '../Data/22_cnn_dataset/test_set',
        target_size=(64, 64),               #Should be same as the input_shape supplied in Conv2D function; GPU => 256, 256
        batch_size=32,                      #GPU => 128 / 256 ...
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,               #Number of images in the Training_set
        epochs=5,                           #GPU => 50 / 100 .....
        validation_data=test_set,
        validation_steps=2000)              #Number of images in the Test_set

