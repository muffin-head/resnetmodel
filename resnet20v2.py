#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 01:46:31 2020

@author: tanmay
"""
#%%
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten,Input,add,Activation,Dense,BatchNormalization,Conv2D,AveragePooling2D
import numpy as np
import math
import os
#%%
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
input_shape=x_train.shape[1:]
num_classes=10
n=3
depth=n*9 +2
version=2
model_type = 'ResNet%dv%d' % (depth, version)
#%%
#normalize the data
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
#%%
#subtract mean from the training data
subtract_mean=True
if subtract_mean:
    x_train_mean=np.mean(x_train,axis=0)
    x_train-=x_train_mean
    x_test-=x_train_mean
#%%
#using one hot encoder on y_train and y_test
y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)
#%%
def lr_schedule(epoch):
    lr=1e-3
    if epoch>180:
        lr*=0.5e-3
    if lr>160:
        lr*=1e-3
    if lr>120:
        lr*=1e-2
    if lr>80:
        lr*=1e-1
    return lr
#%%
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
def resnt_layer(inputs,
                num_filters=16,strides=1,activation='relu',
                kernel_size=3,batch_normalization=True,convfirst=True):
    conv=Conv2D(filters=num_filters,kernel_size=kernel_size,strides=strides,padding='same',
             kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))
    x=inputs
    if convfirst:
        x=conv(x)
        if batch_normalization:
            x=BatchNormalization()(x)
        if activation is not None:
            x=Activation(activation)(x)
    else:
        if batch_normalization:
            x=BatchNormalization()(x)
        if activation is not None:
            x=Activation(activation)(x)
        x=conv(x)
    return x
#%%
def resnet_v2(input_shape,depth,num_classes=10):
    num_filters_in=16
    num_res_block=int((depth-2)/9)
    inputs=Input(shape=input_shape)
    x=resnt_layer(inputs=inputs,num_filters=num_filters_in,convfirst=True)
    for i in range(3):
        for j in range(num_res_block):
            activation='relu'
            strides=1
            batch_normalization=True
            if i==0:
                num_filters_out=num_filters_in*4
                if j==0:
                    batch_normalization=False
                    activation=None
            else:
                num_filters_out=num_filters_in*2
                if j==0:
                    strides=2
            y=resnt_layer(inputs=x,num_filters=num_filters_in,strides=strides,activation=activation,
                          batch_normalization=batch_normalization,convfirst=False,kernel_size=1)
            y=resnt_layer(inputs=y,num_filters=num_filters_in,convfirst=False)
            y=resnt_layer(inputs=y,num_filters=num_filters_out,kernel_size=1,
                          convfirst=False)
            if j==0:
                x=resnt_layer(inputs=x,num_filters=num_filters_out,strides=strides,kernel_size=1,
                          activation=None,batch_normalization=batch_normalization)
            x=add([x,y])
        num_filters_in=num_filters_out
        
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=AveragePooling2D(pool_size=8)(x)
    y=Flatten()(x)
    outputs=Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(y)
    model=Model(inputs=inputs,outputs=outputs)
    return model
#%%
from tensorflow.keras.optimizers import Adam
model = resnet_v2(input_shape=input_shape, depth=depth)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['acc'])
model.summary()
#%%
from tensorflow.keras.callbacks import ReduceLROnPlateau,LearningRateScheduler,ModelCheckpoint

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]
#%%
batch_size = 32
epochs = 200
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
#%%
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
