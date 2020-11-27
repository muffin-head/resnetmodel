#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:31:04 2020

@author: tanmay
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense,Conv2D,Conv2DTranspose,Input,Flatten,Reshape
from tensorflow.keras.layers import LeakyReLU,BatchNormalization,Activation
from tensorflow.keras.models import Model

#%%
def build_gen(inputs,image_size):
    image_resize=image_size//4
    kernel_size=5
    layer_filters=[128,64,32,1]
    x=Dense(image_resize*image_resize*layer_filters[0])(inputs)
    x=Reshape((image_resize,image_resize,layer_filters[0]))(x)
    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides=2
        else:
            strides=1
        x=BatchNormalization()(x)
        X=Activation('relu')(x)
        x=Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)
    x=Activation('sigmoid')(x)
    generator=Model(inputs,x,name='generator')
    return generator
#%%
def build_dis(inputs):
    kernel_size=5
    layer_filters=[32,64,128,256]
    x=inputs
    for filters in layer_filters:
        if filters> layer_filters[-1]:
            strides=1
        else:
            strides=2
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2D(filters=filters, kernel_size=kernel_size,padding='same',strides=strides)(x)
    x=Flatten()(x)
    x=Dense(1)(x)
    x=Activation('sigmoid')(x)
    discriminator=Model(inputs,x,name='discriminator')
    return discriminator
#%%
def train(models,x_train,params):
    generator,discriminator,adversial=models
    batch_size,latent_size,train_steps,model_name=params
    save_interval=500
    noise_input=np.random.uniform(-1.0,1.0,size=[16,latent_size])
    train_size=x_train.shape[0]
    for i in range(train_steps):
        rand_indexes=np.random.randint(0,train_size,size=batch_size)
        real_images=x_train[rand_indexes]
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        fake_images=generator.predict(noise)
        x=np.concatenate((real_images,fake_images))
        
        
#%%
a=np.ones([2*5,1])
print(a)
