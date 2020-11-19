#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 04:09:51 2020

@author: tanmay
"""
#%%
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Conv2DTranspose,Reshape,Input
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras import backend as k
#%%
(x_train,_),(x_test,_)=mnist.load_data()
imagesize=x_train.shape[1]
x_train=np.reshape(x_train,[-1,imagesize,imagesize,1])
x_test=np.reshape(x_test,[-1,imagesize,imagesize,1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
#%%
input_shape=(imagesize,imagesize,1)
kernel_size=3
batch=32
latent_dim=16
layer_filter=[32,64]
#%%
inputs=Input(shape=input_shape,name='encoder_input')
x=inputs
for i in layer_filter:
    x=Conv2D(filters=i, kernel_size=kernel_size,activation='relu',strides=2,padding='same')(x)
shape=k.int_shape(x)
print(shape)
#%%
x=Flatten()(x)
latent=Dense(latent_dim,name="latent_vector")(x)
encoder=Model(inputs,latent,name="encoder")
encoder.summary()
#%%
latent_inputs=Input(shape=(latent_dim,),name='decoder_input')
x=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x=Reshape((shape[1],shape[2],shape[3]))(x)
layer2_filters=[64,32]
for j in layer2_filters:
    x=Conv2DTranspose(filters=i, kernel_size=kernel_size,activation='relu',strides=2,padding='same')(x)
outputs=Conv2DTranspose(filters=1,activation='sigmoid',kernel_size=kernel_size,padding='same',name="decoder_output")(x)
decoder=Model(latent_inputs,outputs,name='decoder')
decoder.summary()
#%%
autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()
#%%
autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=1,
                batch_size=batch)
#%%
x_decoded = autoencoder.predict(x_test)
