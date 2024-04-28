# -*- coding: utf-8 -*-
# """
# Created on Jun 28, 2024
# @author: VIVEK OOMMEN
# """

import os
import sys
import numpy as np
import tensorflow as tf
import scipy.io as io

import time
import matplotlib.pyplot as plt

from unet_lp import Unet

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
      # Invalid device or cannot modify virtual devices once initialized.
        pass

@tf.function()
def train_step(model, x,dt, y, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x,dt)
        loss   = model.Loss(y_pred, y)[0]
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return(loss)

def main():

    np.random.seed(23)
    tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32) 

    d = io.loadmat("../../data/low_pressure_ratio/training_dataset.mat")
    x_train, u_train, v_train = np.array(d['x']), np.array(d['utrain']), np.array(d['vtrain'])

    d = io.loadmat("../../data/low_pressure_ratio/testing_dataset.mat")
    x_test, u_test, v_test = np.array(d['x']), np.array(d['utest']), np.array(d['vtest'])

    Par={}
 
    address = 'unet'
    Par['address'] = address

    Par['nx'] = u_train.shape[1]
    Par['nf'] = u_train.shape[2]
    Par['nt'] = v_train.shape[0]

    step_size = 2
    idx = list(range(0, u_train.shape[0],step_size)) + [u_train.shape[0]-1]
    idx = np.array(idx)
    print('idx:\n', idx)
    print('len(idx): ', len(idx), '\n')

    Par['ns'] = len(idx)

    x = u_train[idx].transpose(1,0,2).reshape(1,Par['nx'], Par['ns']*Par['nf'] )

    x_train = x
    x_test  = x

    X_loc_train = v_train
    X_loc_test  = v_test

    y_train = u_train.reshape(1, -1, Par['nx'], Par['nf'])
    y_test  = u_test.reshape(1, -1, Par['nx'], Par['nf'])

    print('x_train: ', x_train.shape)
    print('X_loc_train: ', X_loc_train.shape)
    print('y_train: ', y_train.shape)
    print()
    print('x_test: ', x_test.shape)
    print('X_loc_test: ', X_loc_test.shape)
    print('y_test: ', y_test.shape)

    MIN = np.min(y_train, axis=(0,1,2))
    MAX = np.max(y_train, axis=(0,1,2))

    Par['inp_shift'] = MIN
    Par['inp_scale'] = MAX - MIN
    Par['out_shift'] = MIN
    Par['out_scale'] = MAX - MIN

    num_samples = x_train.shape[0]

    model = Unet(Par)
    _ = model( tensor(x_train[0:1]), tensor(X_loc_train))
    
    print(model.summary())
    print('Model created')

    model_number = 0
    # UNCOMMENT TO CONTINUE TRAINING FROM PREVIOUS CHECKPOINT
    # model_address = Par['address']+'/best_model.weights.h5' 
    # model.load_weights(model_address)
    
    n_epochs = 100 #Change
    batch_size = 1 
    num_samples = x_train.shape[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-4)   
  
    lowest_loss = 1000
    begin_time = time.time()
    print('Training Begins')
    best_model_id = model_number

    os.makedirs(Par["address"], exist_ok=True)

    for i in range(model_number+1, n_epochs+1):
        
        for j in np.arange(0, num_samples-batch_size+1, batch_size):
            loss = train_step(model, tensor(x_train[j:(j+batch_size)]), tensor(X_loc_train), tensor(y_train[j:(j+batch_size)]), optimizer)

        if i%1 == 0:
            train_loss = loss.numpy()
           
            y_pred = model(x_test,X_loc_test)
            val_loss = tf.reduce_mean( tf.square(y_test - y_pred) ) 
            val_loss = val_loss.numpy()

            if val_loss<lowest_loss:                                                                                                                                                        
                lowest_loss = val_loss
                model.save_weights(address + "/best_model.weights.h5")
                best_model_id = i
 
            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +", best model: "+str(best_model_id) + ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )
        
            model.index_list.append(i)
            model.train_loss_list.append(train_loss)
            model.val_loss_list.append(val_loss)
    
    print('Training complete')

    #Convergence plot
    index_list = model.index_list
    train_loss_list = model.train_loss_list
    val_loss_list = model.val_loss_list
    np.savez(address+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)

    plt.close()
    fig = plt.figure(figsize=(10,7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig( address + "/convergence.png", dpi=800)
    plt.close()
    print('--------Complete--------')      
    


main()
    
