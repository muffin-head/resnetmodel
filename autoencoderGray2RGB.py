#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 04:31:06 2020

@author: tanmay
"""

#%%
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Reshape,Dense,Conv2D,Conv2DTranspose,Flatten
import numpy as np
#%%
import os
(x_train,_),(x_test,_)=cifar10.load_data()
img_rows=x_train.shape[1]
img_cols=x_train.shape[2]
channels=x_train.shape[3]
imgs_dir="savedimg"
savedir=os.path.join(os.getcwd(),imgs_dir)
if not os.path.isdir(savedir):
    os.mkdir(savedir)
#%%
import matplotlib.pyplot as plt
imgs = x_test[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

#%%

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299, 0.587, 0.114])
#%%
'''
print(x_train.shape[:])
a=np.dot(x_train[...,:3],[0.299, 0.587, 0.114])
print("hii")

print(a.shape[:])'''
#%%

'''
import cv2
def rgb2gray(rgb):
    x_train_grayscale = np.zeros(x_train.shape[:-1])
    for i in range(x_train.shape[0]): 
        x_train_grayscale[i] = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY) 
    return x_train_grayscale'''
#%%
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

# display grayscale version of test images
imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.show()
#%%
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train_gray=x_train_gray.astype('float32')/255
x_test_gray=x_test_gray.astype('float32')/255
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)
#%%
from tensorflow.keras import backend as k
input_shape=(img_rows,img_cols,1)
batch_size=32
kernel_size=3
latent_dim=256
latent_filters=[64,128,256]
inputs=Input(shape=input_shape,name="encoder_input")
x=inputs
for i in latent_filters:
    x=Conv2D(filters=i, kernel_size=kernel_size,padding='same',strides=2,activation='relu')(x)
shape=k.int_shape(x)
x=Flatten()(x)
latent=Dense(latent_dim,name='latent_vec')(x)
encoder=Model(inputs,latent,name='encoder')
encoder.summary()
#%%
layer_filters_2=[256,128,64]
latent_inputs=Input(shape=(latent_dim,),name='decoder_input')
x=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x=Reshape((shape[1],shape[2],shape[3]))(x)
for j in layer_filters_2:
    x=Conv2DTranspose(filters=j, kernel_size=kernel_size,padding='same',strides=2,activation='relu')(x)
#outputs=Conv2DTranspose(filters=channels,kernel_size=kernel_size,activation='sigmoid', name="decoder_output")(x)
outputs = Conv2DTranspose(filters=channels,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          strides=2,
                          name='decoder_output')(x)
decoder=Model(latent_inputs,outputs,name="decoder")
decoder.summary()
#%%
autoencoder=Model(inputs,decoder(encoder(inputs)),name="autoencoder")
autoencoder.summary()
#%%
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
autoencoder.compile(loss='mse', optimizer='adam')
callbacks = [lr_reducer, checkpoint]
autoencoder.fit(x_train_gray,
                x_train,
                validation_data=(x_test_gray, x_test),
                epochs=30,
                batch_size=batch_size,
                callbacks=callbacks)
#%%
x_decoded = autoencoder.predict(x_test_gray)
imgs = x_decoded[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/colorized.png' % imgs_dir)
plt.show()
