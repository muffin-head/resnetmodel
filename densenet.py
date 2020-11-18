#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 04:51:56 2020

@author: tanmay
"""
#%%
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Activation, Flatten,concatenate,Conv2D,AveragePooling2D,BatchNormalization,Input
import numpy as np
import os
import math
#%%
num_classes=10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
input_shape=x_train.shape[1:]
#normalize the data
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)
#%%
depth=100
num_dense_blocks=3
growth=12
num_bottleneck_layer=(depth-4)//(2*num_dense_blocks)
filters_before_dense=2*growth
compression=0.5
#%%
def lr_schedule(epoch):
    lr=1e-3
    if epoch>180:
        lr*=0.5e-3
    if epoch>160:
        lr*=1e-3
    if epoch>120:
        lr*=1e-2
    if epoch>80:
        lr*=1e-1
    return lr
#%%
inputs=Input(shape=input_shape)
x=BatchNormalization()(inputs)
x=Activation('relu')(x)
x=Conv2D(filters=filters_before_dense,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
x=concatenate([inputs,x])
for i in range(num_dense_blocks):
    for j in range(num_bottleneck_layer):
        y=BatchNormalization()(x)
        y=Activation('relu')(y)
        y=Conv2D(filters=4*growth, kernel_size=1,kernel_initializer='he_normal',padding='same')(y)
        x=concatenate([x,y])
    if i==num_dense_blocks-1:
        continue
    y=BatchNormalization()(x)
    filters_before_dense+=growth*num_bottleneck_layer
    filters_before_dense=int(filters_before_dense*compression)
    y=Conv2D(filters=filters_before_dense,kernel_initializer='he_normal',kernel_size=1,padding='same')(y)
    x=AveragePooling2D()(y)
x=AveragePooling2D(pool_size=8)(x)
y=Flatten()(x)
outputs=Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(y)
model=Model(inputs=inputs,outputs=outputs)
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(1e-3),
              metrics=['acc'])
model.summary()

#%%
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_densenet_model.{epoch:02d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# prepare callbacks for model saving and for learning rate reducer
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
scores = model.evaluate(x_test, y_test, verbose=0)