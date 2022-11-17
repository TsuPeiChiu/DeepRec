import os
import numpy as np
import random as ra
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, \
                        Flatten, Dense, Dropout, Reshape, Lambda, UpSampling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras import optimizers as op
import deeprec.metrics as mt

from keras.layers.normalization import BatchNormalization

def build_deeprec_model(params, seq_len=10, random_state=None, mode='pc'):
    """ """
    if random_state != None:
        os.environ['PYTHONHASHSEED']=str(random_state)
        np.random.seed(random_state)
        ra.seed(random_state)
    
    # build convolutional layers
#    patch_size = params.hbond_major['filter_len']
    patch_size = 10

    # construct a net
    if mode=='seq':
        model = __construct_seq_net(params, seq_len, patch_size)
    else:
        model = __construct_pc_net(params, seq_len, patch_size)
   
    return model


def __construct_seq_net(params, seq_len, patch_size):
    # construct 1D input layer
    channel, height, length = 4, 1, (2*seq_len+patch_size)
    input_data = Input(shape=(channel*height*length,), name="seq_x")
    input_reshape = Reshape((channel,height,length))(input_data)

    # construct conv2D layer
    conv_seq = Conv2D(filters=params.onehot_seq['nb_filter_1'],
                        kernel_size=(params.onehot_seq['filter_hei_1'],
                                     params.onehot_seq['filter_len_1']),
                        activation=params.onehot_seq['activation_1'],
                        kernel_initializer='glorot_uniform',
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=regularizers.l1_l2(
                            l1=params.onehot_seq['alpha_1']*params.onehot_seq['l1_ratio_1'],
                            l2=params.onehot_seq['alpha_1']*(1-params.onehot_seq['l1_ratio_1']))
                        )(input_reshape)
    pool_seq = MaxPooling2D(pool_size=(params.onehot_seq['pool_hei_1'],
                                params.onehot_seq['pool_len_1']))(conv_seq)

    batch_seq = BatchNormalization()(pool_seq)


    flat_seq = Flatten()(batch_seq)

    # fully connected and output layers
    #hidden = Dense(units=params.joint['nb_hidden'],
    #                activation=params.joint['activation'],
    #                kernel_initializer='glorot_uniform',
    #                kernel_regularizer=regularizers.l1_l2(
    #                    l1=params.joint['alpha']*params.joint['l1_ratio'],
    #                    l2=params.joint['alpha']*(1-params.joint['l1_ratio']))
    #                )(flat_seq)
    #dropout = Dropout(params.joint['drop_out'], noise_shape=None)(hidden)
    #output = Dense(1, activation=params.target['activation'])(dropout)
    output = Dense(1, activation=params.target['activation'])(flat_seq)


    # summarize the model
    model = Model(inputs=input_data, outputs=output)
#    model.summary()
    model.compile(optimizer=op.Adam(lr=params.optimizer_params['lr']),
                  loss=params.loss, metrics=[mt.r_squared])

    return model
    
def __construct_pc_net(params, seq_len, patch_size):
    # construct 1D input layer
    channel, height, length = 4, 7, (2*seq_len+patch_size)
    input_data = Input(shape=(channel*height*length,), name="hbond_x")

    # construct conv2D layer
    input_reshape = Reshape((channel,height,length))(input_data)
    input_split = Lambda(lambda x:tf.split(x, [4,3], axis=2))(input_reshape)
    input_major = input_split[0]
    input_minor = input_split[1]

    # conv2D for major
    conv_major = Conv2D(filters=params.hbond_major['nb_filter_1'], 
                        kernel_size=(params.hbond_major['filter_hei_1'], 
                                     params.hbond_major['filter_len_1']),
                        activation=params.hbond_major['activation_1'],
                        kernel_initializer='glorot_uniform',                                                                
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=regularizers.l1_l2(
                            l1=params.hbond_major['alpha_1']*params.hbond_major['l1_ratio_1'],
                            l2=params.hbond_major['alpha_1']*(1-params.hbond_major['l1_ratio_1']))
                        )(input_major)
    pool_major = MaxPooling2D(pool_size=(params.hbond_major['pool_hei_1'],
                                params.hbond_major['pool_len_1']))(conv_major)
    
    conv_major = Conv2D(filters=params.hbond_major['nb_filter_2'], 
                        kernel_size=(params.hbond_major['filter_hei_2'],
                                     params.hbond_major['filter_len_2']),
                        activation=params.hbond_major['activation_2'],
                        kernel_initializer='glorot_uniform',                                                                
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=regularizers.l1_l2(
                                l1=params.hbond_major['alpha_2']*params.hbond_major['l1_ratio_2'],
                                l2=params.hbond_major['alpha_2']*(1-params.hbond_major['l1_ratio_2']))
                        )(pool_major)

    pool_major = MaxPooling2D(pool_size=(params.hbond_major['pool_hei_2'],
                                params.hbond_major['pool_len_2']))(conv_major)
    
    flat_major = Flatten()(pool_major)

    batch_major = BatchNormalization()(flat_major)
   
    # conv2D for minor
    conv_minor = Conv2D(filters=params.hbond_minor['nb_filter_1'], 
                        kernel_size=(params.hbond_minor['filter_hei_1'], 
                                     params.hbond_minor['filter_len_1']),
                        activation=params.hbond_minor['activation_1'],
                        kernel_initializer='glorot_uniform',                                                                
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=regularizers.l1_l2(
                                l1=params.hbond_minor['alpha_1']*params.hbond_minor['l1_ratio_1'],
                                l2=params.hbond_minor['alpha_1']*(1-params.hbond_minor['l1_ratio_1']))
                        )(input_minor)
    pool_minor = MaxPooling2D(pool_size=(params.hbond_minor['pool_hei_1'],
                                params.hbond_minor['pool_len_1']))(conv_minor)
    
    conv_minor = Conv2D(filters=params.hbond_minor['nb_filter_2'], 
                        kernel_size=(params.hbond_minor['filter_hei_2'],
                                     params.hbond_minor['filter_len_2']),
                        activation=params.hbond_minor['activation_2'],
                        kernel_initializer='glorot_uniform',                                                                       
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=regularizers.l1_l2(
                                l1=params.hbond_minor['alpha_2']*params.hbond_minor['l1_ratio_2'],
                                l2=params.hbond_minor['alpha_2']*(1-params.hbond_minor['l1_ratio_2']))
                        )(pool_minor)
    pool_minor = MaxPooling2D(pool_size=(params.hbond_minor['pool_hei_2'],
                                params.hbond_minor['pool_len_2']))(conv_minor)    
    
    flat_minor = Flatten()(pool_minor)
    
    batch_minor = BatchNormalization()(flat_minor)


    # Construct the hidden layer
#    merge = concatenate([batch_major, batch_minor])
    merge = concatenate([flat_major, flat_minor])
    hidden = Dense(units=params.joint['nb_hidden'], 
                   activation=params.joint['activation'], 
                   kernel_regularizer=regularizers.l1_l2(
                           l1=params.joint['alpha']*params.joint['l1_ratio'], 
                           l2=params.joint['alpha']*(1-params.joint['l1_ratio']))
                   )(merge)
    dropout = Dropout(params.joint['drop_out'], noise_shape=None)(hidden)
    output = Dense(1, activation=params.target['activation'])(dropout)

    # summarize the model
    model = Model(inputs=input_data, outputs=output)
#    model.summary()
    model.compile(optimizer=op.Adam(lr=params.optimizer_params['lr']), 
                  loss=params.loss, metrics=[mt.r_squared])
        
    return model


def build_transfer_model(params, model):
    """ """
    model.summary()
    
    # remove dropout and output layers
    #model.layers.pop()
    #model.layers.pop()

    # add a new dense, dropout, and output layers
    #new_dense = Dense(units=32, activation='softmax', 
    #        name='new_dense')(model.layers[-1].output)

    #new_dropout = Dropout(0.1, noise_shape=None)(new_dense)
    #out = Dense(1, activation='relu')(new_dropout)
    #inp = model.input
    #model = Model(inp, out)

    # freeze the pre-trained model
    #for layer in model.layers[:5]:
    #for layer in model.layers[:15]:
    #for layer in model.layers[:5]:
    #    layer.trainable=False
    #model.summary()

    # compile model
    model.compile(optimizer=op.Adam(lr=params.optimizer_params['lr']),
                  loss=params.loss, metrics=[mt.r_squared])

    return model


def build_autoencoder_model(params, seq_len=10, random_state=None):
    """ """
    if random_state != None:
        os.environ['PYTHONHASHSEED']=str(random_state)
        np.random.seed(random_state)
        ra.seed(random_state)
#    keras.backend.clear_session()
    
    # build convolutional layers
#    patch_size = params.hbond_major['filter_len']
    patch_size = 10

    # construct the 1D input layer
    channel, height, length = 4, 4, (2*seq_len+patch_size)
    input_major = Input(shape=(channel,height,length,), name="hbond_x")
    
    
    x = Conv2D(16, (2, 4), activation='relu', padding='same', data_format='channels_first')(input_major)
    x = MaxPooling2D((1, 1), padding='same')(x)
    x = Conv2D(16, (2, 4), activation='relu', padding='same', data_format='channels_first')(x)
    x = MaxPooling2D((1, 1), padding='same')(x)
    x = Conv2D(16, (2, 4), activation='relu', padding='same', data_format='channels_first')(x)
    encoded = MaxPooling2D((1, 1), padding='same')(x)

    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    

    x = Conv2D(16, (2, 4), activation='relu', padding='same', data_format='channels_first')(encoded)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(16, (2, 4), activation='relu', padding='same', data_format='channels_first')(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(16, (2, 4), activation='relu', padding='same', data_format='channels_first')(x)
    x = UpSampling2D((1, 1))(x)
    
    decoded = Conv2D(4, (2, 4), activation='sigmoid', padding='same', data_format='channels_first')(x)
    
    autoencoder = keras.Model(input_major, decoded)
    autoencoder.summary()
    
    autoencoder.compile(optimizer=op.Adam(lr=0.0001), loss=params.loss)
    
    return autoencoder
    



























    
