#%%
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input, BatchNormalization, Activation, AveragePooling2D,Conv2D,add,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import os
import math
#%%
#loading the data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
#getting the input shape
#%%
input_shape=x_train.shape[1:]
print(input_shape)
#normalize the data
#%%
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
print(x_train[0])
#%%
#subtracting pixel for accc
subtract_pixel_mean=True
if subtract_pixel_mean:
    x_train_mean=np.mean(x_train,axis=0)
    x_train-=x_train_mean
    x_test-=x_train_mean
#%%
#converting y var to categorical using one hot encoder
num_classes=10
y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)
#%%
n=3
version=1
depth=n*6+2
model_type="resnet%dv%d"%(depth,version)
batch_size=32
epochs=200
data_augmentation=True
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
    print("learning rate:",lr)
    return lr
#%%
def resnet_layer(inputs,
                 num_filters=16,
                 activation="relu",
                 kernel_size=3,
                 strides=1,
                 batch_normalization=True,
                 conv_first=True):
    conv=Conv2D(num_filters,padding='same',kernel_initializer="he_normal",kernel_size=kernel_size,
                kernel_regularizer=l2(1e-4),strides=strides)
    x=inputs
    if conv_first:
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
def resnetV1(input_shape,depth,num_classes=10):
    num_filt=16
    num_res_block=int((depth-2)/6)
    inputs=Input(shape=input_shape)
    x=resnet_layer(inputs=inputs)
    for i in range(3):
        for j in range(num_res_block):
            strides=1
            if i>0 and j==0:
                strides=2
            y=resnet_layer(inputs=x,num_filters=num_filt,strides=strides)
            y=resnet_layer(inputs=y,num_filters=num_filt,strides=strides,activation=None)
            if i>0 and j==0:
                x=resnet_layer(inputs=x,num_filters=num_filt,activation=None
                               ,batch_normalization=False,kernel_size=1,strides=strides)
        x=add([x,y])
        x=Activation("relu")(x)
    x=AveragePooling2D(pool_size=8)(x)
    y=Flatten()(x)
    outputs=Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(y)
    model=Model(inputs=inputs,outputs=outputs)
    return model
#%%
model=resnetV1(input_shape=input_shape, depth=depth)
model.compile(optimizer=Adam(lr=lr_schedule(0)),loss='categorical_crossentropy',metrics=['acc'])
model.summary()
#%%
from tensorflow.keras.callbacks import ModelCheckpoint
filepath='/media/tanmay/F'
model_name='cifar10_%s_model.{epoch:03d}.h5' % model_type
fp=os.path.join(filepath,model_name)
checkpoint=ModelCheckpoint(filepath=fp,monitor='val_acc',verbose=1,save_best_only=True)
lr_scheduler=LearningRateScheduler(lr_schedule)
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
call_backs=[checkpoint,lr_reducer,lr_scheduler]
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
                           samplewise_std_normalization=False,zca_whitening=False,rotation_range=0,
                           width_shift_range=0.1,height_shift_range=0.1,
                           horizontal_flip=True,vertical_flip=False)
datagen.fit(x_train)
steps_per_epoch=math.ceil(len(x_train)/batch_size)
model.fit(x=datagen.flow(x_train,y_train,batch_size=batch_size),verbose=1,epochs=epochs,
          validation_data=(x_test,y_test),steps_per_epoch=steps_per_epoch,callbacks=call_backs)
#%%
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
    
    
    
    
    
    
    
    
    