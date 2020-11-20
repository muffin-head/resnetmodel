#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 02:12:03 2020

@author: tanmay
"""

#%%
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,Conv2DTranspose,Input,Reshape,Flatten
from tensorflow.keras import backend as k
import numpy as np

#%%
(x_train,_),(x_test,_)=mnist.load_data()
image_size=x_train.shape[1]
x_train=np.reshape(x_train,[-1,image_size,image_size,1])
x_test=np.reshape(x_test, [-1,image_size,image_size,1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
#%%
noise=np.random.normal(loc=0.5,scale=0.5,size=x_train.shape)
x_train_noisy=x_train +noise
noise=np.random.normal(loc=0.5,scale=0.5,size=x_test.shape)
x_test_noisy=x_test +noise
x_train_noisy=np.clip(x_train_noisy,0.,1.)
x_test_noisy=np.clip(x_test_noisy,0.,1.)
#%%
input_shape=(image_size,image_size,1)
batch_size=32
latent_dim=16
kernel_size=3
layer_filters=[32,64]
inputs=Input(shape=input_shape,name="encoder_input")
x=inputs
for i in layer_filters:
    x=Conv2D(filters=i,kernel_size=kernel_size,padding='same',strides=2,activation='relu')(x)
shape=k.int_shape(x)
print(shape)
x= Flatten()(x)
latent=Dense(latent_dim,name="decoder_input")(x)
encoder=Model(inputs,latent,name="encoder")
encoder.summary()
#%%
inputs2=Input(shape=(latent_dim,), name="decoder_input")
x=Dense(shape[1]*shape[2]*shape[3])(inputs2)
x=Reshape((shape[1],shape[2],shape[3]))(x)

layer_second_filters=[64,32]
for j in layer_second_filters:
    x=Conv2DTranspose(filters=j, kernel_size=kernel_size,strides=2,padding='same',activation='relu')(x)
outputs=Conv2DTranspose(filters=1, kernel_size=kernel_size,padding='same',activation='sigmoid',name='decoder_output')(x)
decoder=Model(inputs2,outputs,name="decoder")
decoder.summary()
#%%
autoencoder=Model(inputs,decoder(encoder(inputs)),name="autoencoder")
autoencoder.summary()
#%%
autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=10,
                batch_size=batch_size)
#%%
import matplotlib.pyplot as plt
from PIL import Image

rows, cols = 3, 9
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
print(imgs.shape[:])
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()
