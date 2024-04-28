# -*- coding: utf-8 -*-
# """
# Created on Sat Jun  5 21:47:39 2021

# @author: VIVEK OOMMEN
# """

import numpy as np
import tensorflow as tf
print('tf version: ', tf.__version__)
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Input, Reshape, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D,  PReLU, Flatten, Dense, Activation, LayerNormalization, GroupNormalization
from tensorflow.keras.losses import MeanSquaredError


import matplotlib
import matplotlib.pyplot as plt
import time

import os

class Unet(tf.keras.Model):
    def __init__(self, Par):
        super(Unet, self).__init__()
        np.random.seed(23)        
        tf.random.set_seed(23)

        #Defining some model parameters
        self.Par = Par
        
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        
        self.lr=10**-4

        n_kernels = 32
        kx=3
        ky=3
        activation='gelu'
        padding='same'
        
        #Defining the unet layers
        #self.enc1 = self.block(n_kernels, kx,ky, activation, padding, name='conv1', flag=True)                               #[_,128,128, n_kernels]
        #self.pool1= MaxPooling2D(pool_size = (2,2), strides = 2, name='pool1')                                               #[_,64,64, n_kernels]
        self.enc2 = self.block(2*n_kernels, kx, activation, padding, name='conv2', flag=True)                             #[_,64,64, 2*n_kernels]
        self.pool2= MaxPooling1D(pool_size = 2, strides = 2, name='pool2')                                                #[_,32,32, 2*n_kernels]
        self.enc3 = self.block(4*n_kernels, kx, activation, padding, name='conv3')                                        #[_,32,32, 4*n_kernels]
        self.pool3= MaxPooling1D(pool_size = 2, strides = 2, name='pool3')                                                #[_,16,16, 4*n_kernels]
        self.enc4 = self.block(8*n_kernels, kx, activation, padding, name='conv4')                                        #[_,16,16, 8*n_kernels]
        self.pool4= MaxPooling1D(pool_size = 2, strides = 2, name='pool4')                                                #[_,8,8, 8*n_kernels] 

        self.bottleneck = self.block(16*n_kernels, kx, activation, padding, name='bottleneck')                            #[_,8,8, 16*n_kernels]

        self.tconv4 = Conv1DTranspose(8*n_kernels, kernel_size=2, strides=2, activation = 'gelu',  name='tconv4')         #[_,16,16, 8*n_kernels]
        self.dec4   = self.block(8*n_kernels, kx, activation, padding, name='dec4')                                       #[_,16,16, 8*n_kernels]
        self.tconv3 = Conv1DTranspose(4*n_kernels, kernel_size=2, strides=2, activation = 'gelu',  name='tconv3')         #[_,32,32, 4*n_kernels]
        self.dec3   = self.block(4*n_kernels, kx, activation, padding, name='dec3')                                       #[_,32,32, 4*n_kernels]
        self.tconv2 = Conv1DTranspose(2*n_kernels, kernel_size=2, strides=2, activation = 'gelu',  name='tconv2')         #[_,64,64, 2*n_kernels]
        self.dec2   = self.block(2*n_kernels, kx, activation, padding, name='dec2')                                       #[_,64,64, 2*n_kernels]
        #self.tconv1 = Conv2DTranspose(1*n_kernels, kernel_size=(2,2), strides=(2,2), activation = 'gelu',  name='tconv1')    #[_,128,128, 1*n_kernels]
        #self.dec1   = self.block(1*n_kernels, kx,ky, activation, padding, name='dec1')                                       #[_,128,128, 1*n_kernels]
        
        self.final_norm = GroupNormalization( int(n_kernels/4) )
        self.final  = Conv1D(self.Par['nf'], 1, activation='relu')                                                     #[_,128,128, self.nf]   

        #Defining the trunk network
        self.trunk_net = Sequential(name='trunk_net')
        self.trunk_net.add(Dense(128))
        self.trunk_net.add(Activation(tf.math.sin))
        self.trunk_net.add(Dense(128))
        self.trunk_net.add(Activation(tf.math.sin))

        self.dense1 = Dense(1*n_kernels)
        self.dense2 = Dense(2*n_kernels)
        self.dense3 = Dense(4*n_kernels)
        self.dense4 = Dense(8*n_kernels)
        self.dense5 = Dense(16*n_kernels)

    # Q: ordering of conv, norm, activation
    def block(self, n_kernels, kx, activation, padding, name, flag=False, n_groups=1):
        block = Sequential(name=name)
        if flag:
            block.add( Conv1D(n_kernels, kx, padding=padding, input_shape=[self.Par['nx'], self.Par['ns']*self.Par['nf']  ] ) )
        else:
            block.add( Conv1D(n_kernels, kx, padding=padding ) )

        block.add( GroupNormalization(n_groups) )
        block.add( Activation(activation) )

        return block

    @tf.function()
    def call(self, x, dt):
    # x - [_,128,128,lb,nf]
    # dt- [nt,1]
        nt = tf.shape(dt)[0]        

        x = tf.reshape(x, [1, self.Par['nx'], self.Par['ns'], self.Par['nf'] ] )
        
        rho = (self.logging_fn(x[:,:,:,0:1]) - self.Par['log_shift'][0])/self.Par['log_scale'][0]
        u   = (x[:,:,:,1:2] - self.Par['out_shift'][1])/self.Par['out_scale'][1]
        P   = (self.logging_fn(x[:,:,:,2:3]) - self.Par['log_shift'][2])/self.Par['log_scale'][2]
        x = tf.concat([rho,u,P], axis=-1)
        
        x = tf.reshape(x, [1, self.Par['nx'], self.Par['ns']*self.Par['nf'] ] )

        dt = (dt - self.Par['tn_shift'])/self.Par['tn_scale']

        #e1 = self.enc1(x)                                                                               #[_,128,128, n_kernels]
        e2 = self.enc2(x)                                                                               #[_,64,64, 2*n_kernels]
        e3 = self.enc3(self.pool2(e2))                                                                  #[_,32,32, 4*n_kernels]
        e4 = self.enc4(self.pool3(e3))                                                                  #[_,16,16, 8*n_kernels]

        bottleneck = self.bottleneck(self.pool4(e4))                                                    #[_,8,8, 16*n_kernels]
 
        f_dt = self.trunk_net(dt)                                                                         #[nt, 128]
        f1_dt = self.dense1(f_dt)                                                                         #[nt, 1*n_kernels]
        f2_dt = self.dense2(f_dt)                                                                         #[nt, 2*n_kernels]
        f3_dt = self.dense3(f_dt)                                                                         #[nt, 4*n_kernels]
        f4_dt = self.dense4(f_dt)                                                                         #[nt, 8*n_kernels]
        f5_dt = self.dense5(f_dt)                                                                         #[nt, 16*n_kernels]

        temp = tf.einsum('ijl,pl->ipjl', bottleneck,f5_dt)                                              #[_,nt,8,8,16*n_kernels]
        BOTTLENECK = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3]])                       #[_*nt,8,8,16*n_kernels]

        d4 = self.tconv4(BOTTLENECK)                                                                    #[_*nt,16,16, 8*n_kernels]
        temp = tf.einsum('ijl,pl->ipjl', e4,f4_dt)                                                      #[_,nt,16,16,8*n_kernels]
        E4 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3] ])                              #[_*nt,16,16,8*n_kernels]
        d4 = tf.concat([d4,E4], axis=-1)                                                                #[_*nt,16,16, 2*(8*n_kernels)]
        d4 = self.dec4(d4)                                                                              #[_*nt,16,16, 8*n_kernels]

        d3 = self.tconv3(d4)                                                                            #[_*nt,32,32, 4*n_kernels]
        temp = tf.einsum('ijl,pl->ipjl', e3,f3_dt)                                                      #[_,nt,32,32,4*n_kernels]
        E3 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3] ])                              #[_*nt,32,32,4*n_kernels]
        d3 = tf.concat([d3,E3], axis=-1)                                                                #[_*nt,32,32, 2*(4*n_kernels)]
        d3 = self.dec3(d3)                                                                              #[_*nt,32,32, 4*n_kernels]

        d2 = self.tconv2(d3)                                                                            #[_*nt,64,64, 2*n_kernels]
        temp = tf.einsum('ijl,pl->ipjl', e2,f2_dt)                                                      #[_,nt,64,64,2*n_kernels]
        E2 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3]])                               #[_*nt,64,64,2*n_kernels]
        d2 = tf.concat([d2,E2], axis=-1)                                                                #[_*nt,64,64, 2*(2*n_kernels)]
        d2 = self.dec2(d2)                                                                              #[_*nt,64,64, 2*n_kernels]

        #d1 = self.tconv1(d2)                                                                            #[_*nt,128,128, 1*n_kernels]
        #temp = tf.einsum('ijkl,pl->ipjkl', e1,f1_dt)                                                    #[_,nt,128,128,1*n_kernels]
        #E1 = tf.reshape(temp, [-1, tf.shape(temp)[2], tf.shape(temp)[3], tf.shape(temp)[4]])            #[_*nt,128,128,1*n_kernels]
        #d1 = tf.concat([d1,E1], axis=-1)                                                                #[_*nt,128,128, 2*(1*n_kernels)]
        #d1 = self.dec1(d1)                                                                              #[_*nt,128,128, 1*n_kernels]

        out = self.final_norm(d2)
        out = self.final(out)                                                                           #[_*nt,128,128, self.nf]
        out = tf.reshape(out, [-1,nt,tf.shape(out)[1],tf.shape(out)[2] ])                               #[_,nt,128,128]
        
        rho_out = out[:,:,:,0:1]*self.Par['log_scale'][0] + self.Par['log_shift'][0]
        rho_out = tf.clip_by_value(rho_out, -1*(10**25), 65)
        rho_out = tf.math.exp(rho_out) #- self.Par['eps']
        u_out   = out[:,:,:,1:2]*self.Par['out_scale'][1] + self.Par['out_shift'][1]
        P_out   = out[:,:,:,2:3]*self.Par['log_scale'][2] + self.Par['log_shift'][2]
        P_out = tf.clip_by_value(P_out, -1*(10**25), 65)
        P_out   = tf.math.exp(P_out) #- self.Par['eps']

        final_out = tf.concat([rho_out, u_out, P_out], axis=-1)
       
        return final_out

    def logging_fn(self, x):
        return tf.math.log( x )# + self.Par['eps'])

    @tf.function()
    def Loss(self, y_pred, y_train):

        rho_true = (self.logging_fn(y_train[:,:,:,0]) - self.Par['log_shift'][0])/self.Par['log_scale'][0]
        u_true   = (y_train[:,:,:,1] - self.Par['out_shift'][1])/self.Par['out_scale'][1]
        P_true   = (self.logging_fn(y_train[:,:,:,2]) - self.Par['log_shift'][2])/self.Par['log_scale'][2]

        rho_pred = (self.logging_fn(y_pred[:,:,:,0]) - self.Par['log_shift'][0])/self.Par['log_scale'][0]
        u_pred   = (y_pred[:,:,:,1] - self.Par['out_shift'][1])/self.Par['out_scale'][1]
        P_pred   = (self.logging_fn(y_pred[:,:,:,2]) - self.Par['log_shift'][2])/self.Par['log_scale'][2]

        mse = MeanSquaredError()
        
        #-------------------------------------------------------------#
        #Total Loss
        term1 = mse(rho_true,rho_pred)
        term2 = mse(u_true,u_pred)
        term3 = mse(P_true,P_pred)
        train_loss = term1 + term2 + term3 
        #-------------------------------------------------------------#

        # print(f'term1: {term1.numpy():.3e}, term2: {term2.numpy():.3e}, term3: {term3.numpy():.3e}')
        
        return([train_loss])
        
