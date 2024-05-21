# Block 11
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dropout, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Activation, Conv2DTranspose, BatchNormalization, add, multiply
import tensorflow.keras.backend as K

def Conv2D_block(input_layer, out_n_filters, kernel_size=[3,3], stride=[1,1], padding='same'):
    
    layer = input_layer
    
    for i in range(2):
        
        layer = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)
        layer = BatchNormalization()(layer) 
        layer = Activation('relu')(layer) # ReLU
        
    out_layer = layer
    
    return out_layer

def Up_and_Concate(down_layer, layer):
    
    input_channel = down_layer.get_shape().as_list()[3]
    output_channel = input_channel // 2
    
    up = UpSampling2D(size = (2,2))(down_layer)

    concate = concatenate([up, layer])
    return concate


def classifier(conv_last, n_label, activation):
    x = Conv2D(n_label, kernel_size=[1, 1], padding='same')(conv_last) # 필터셋개수=레이블개수, # 필터셋사이즈=[1,1], 패딩
    output = Activation(activation)(x)  
   
    return output

def dice_coef(y_true, y_pred, smooth=0.001):
    y_true_f = K.flatten(y_true)# y_true를 일렬로 정렬한다.
    y_pred_f = K.flatten(y_pred) # y_pred를 일렬로 정렬한다.
    intersection = K.sum(y_true_f * y_pred_f)# y_true_f와 y_pred_f, 두 이미지를 곱해서 교집합을 구한다.
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def build_model(input_shape, classes, activation='sigmoid'): # image Segmentation을 위한 Convolutional Network End-to-End 방식
        
    inputs = Input(input_shape, dtype = 'float32')
    x = inputs
    depth = 4 # 채널 개수
    features = 32 # 특징 개수
    down_layer = []
    
    
    for i in range(depth):
        
        x = Conv2D_block(x, features) 
        down_layer.append(x)
        x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(x)

        features = features * 2
        
    x = Conv2D_block(x, features)
    
    for i in reversed(range(depth)):

        features = features // 2
        
        x = Up_and_Concate(x, down_layer[i])
        x = Conv2D_block(x, features)
    
    
    output = classifier(x, classes, activation)
    model = Model(inputs = inputs, outputs = output)
    
    return model

