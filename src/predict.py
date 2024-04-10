import jax
import numpy as np
import scipy.io as io
from scipy import signal
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import optax
import pickle
import matplotlib.pyplot as plt
import sys

from jax import grad, jit, vmap, value_and_grad
from jax import random
key = random.PRNGKey(1234)
from jax.example_libraries import optimizers

def infer_solution(inputs):
    num_epochs = inputs['Epochs']['Branch']+1
    lr = inputs['learning_rate']
    nt = inputs['Ensembles']
    scaling = inputs['Data_scaling']
    foldert = inputs['Model_folder']['Trunk']
    folderb = inputs['Model_folder']['Branch']
    Problem = inputs['Problem']
    FSM = inputs['FSM']['Branch']
    FRL = inputs['FRL']['Branch']
    layers_x = inputs['Architecture']['Trunk']
    activation = inputs['RBA']['Branch']
    BasefuncB = getattr(jnp, activation)
    activation = inputs['RBA']['Trunk']
    BasefuncT = getattr(jnp, activation)
    decomposition = inputs['Decomposition']
    Result_folder = inputs['Results_folder']

    command = 'rm -r '+Result_folder
    os.system(command)
    command = 'mkdir '+Result_folder
    os.system(command)

    G_dim = layers_x[-1]

    match Problem:
        case "HPR":
            train_set= "../data/high_pressure_ratio_LeBlanc/training_dataset.mat"
            test_set= "../data/high_pressure_ratio_LeBlanc/testing_dataset.mat"
        case "IPR":
            train_set= "../data/intermediate_pressure_ratio/training_dataset.mat"
            test_set= "../data/intermediate_pressure_ratio/testing_dataset.mat"
        case "LPR":
            train_set= "../data/low_pressure_ratio/training_dataset.mat"
            test_set= "../data/low_pressure_ratio/testing_dataset.mat"

    d = io.loadmat(train_set)
    x_train, u_train, v_train = jnp.array(d['x']), jnp.array(d['utrain']), jnp.array(d['vtrain'])

    d = io.loadmat(test_set)
    x_test, u_test, v_test = jnp.array(d['x']), jnp.array(d['utest']), jnp.array(d['vtest'])

    if Problem == "HPR":
        u_train = u_train.at[:,:,0].set(jnp.log(u_train[:,:,0]))
        u_train = u_train.at[:,:,2].set(jnp.log(u_train[:,:,2]))

        u_test = u_test.at[:,:,0].set(jnp.log(u_test[:,:,0]))
        u_test = u_test.at[:,:,2].set(jnp.log(u_test[:,:,2]))

    Xmin = np.min(v_train)
    Xmax = np.max(v_train)

    dmin = np.zeros((1,1,3))
    dmax = np.zeros((1,1,3))

    fac  = 1e-10*np.ones_like(dmin)

    data = jnp.concatenate([u_train,u_test], axis=0, dtype=None)

    print("data=\t",data.shape)

    dmin[0,0,0] = np.min(u_train[:,:,0])
    dmax[0,0,0] = np.max(u_train[:,:,0])
    dmin[0,0,1] = np.min(u_train[:,:,1])
    dmax[0,0,1] = np.max(u_train[:,:,1])
    dmin[0,0,2] = np.min(u_train[:,:,2])
    dmax[0,0,2] = np.max(u_train[:,:,2])


    oness = np.ones((1,1,3))
    if scaling=='01':
        u_train = (u_train - dmin)/(dmax - dmin)
        u_test = (u_test- dmin)/(dmax - dmin)
        tol = (fac-dmin)/(dmax - dmin)
        v_train = (v_train - Xmin)/(Xmax - Xmin) 
        v_test = (v_test- Xmin)/(Xmax - Xmin) 
    else:
        u_train = 2*(u_train - dmin)/(dmax - dmin) - oness
        u_test = 2*(u_test- dmin)/(dmax - dmin) - oness
        tol = 2*(fac-dmin)/(dmax - dmin) - oness
        v_train = 2.*(v_train - Xmin)/(Xmax - Xmin) - 1.0
        v_test = 2.*(v_test- Xmin)/(Xmax - Xmin) - 1.0

    def load_model(filename):
        with open(filename, 'rb') as file:
            param=pickle.load(file)
        return param


    def fnn_B(X, W, b, a, c, a1, F1, c1):
        inputs = X
        L = len(W)
        for i in range(L-1):
            inputs =  BasefuncB(jnp.add(10*a[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c[i])) \
                + 10*a1[i]*jnp.sin(jnp.add(10*F1[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c1[i])) 
        Y = jnp.dot(inputs, W[-1]) + b[-1]   
        return Y

    def fnn_T(X, W, b, a, c, a1, F1, c1):
        inputs = X
        L = len(W)
        for i in range(L-1):
            inputs =  BasefuncT(jnp.add(10*a[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c[i])) \
                + 10*a1[i]*jnp.sin(jnp.add(10*F1[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c1[i])) 
        Y = jnp.dot(inputs, W[-1]) + b[-1]     
        return Y

    def predictT(params, data):
        Am, W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk = params
        v, x = data
        Am = jnp.reshape(Am,(-1,G_dim,3))
        u_out_trunk = fnn_T(x, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk)
        u_pred = jnp.einsum('imn,jm->ijn',Am, u_out_trunk) # matmul

        return u_pred

    def choose_trunk_model(folder,data,u, tol):
        files = os.listdir(folder)
        f = []
        for i in range(nt):
            temp = []
            for file in files:
                if 'model'+str(i+1)+'.' in file:
                    temp.append(file)
            f.append(temp)
        model = []
        for i in range(nt):
            l2norm=[]
            for j in range(len(f[i])):
                filename = folder+f[i][j]
                print("filename=\t",filename)
                params = load_model(filename)
                pred = predictT(params, data)
                l2 = jnp.mean(jnp.linalg.norm(u - pred, 2, axis=1)/np.linalg.norm(u , 2, axis=1))
                l2norm.append(l2)
            l2norm = jnp.array(l2norm)
            minimum = jnp.argmin(l2norm)
            model.append(f[i][minimum])
        return model
    foldert = './'+foldert+'/'
    modelt = choose_trunk_model(foldert,[v_train, x_train],u_train, tol)
    print("modelt=\t",modelt)

    def predict(paramsb,paramst,Rinv, data, tol):
        Am, W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk = paramst
        W_branch,b_branch,a_branch, c_branch, a1_branch, F1_branch , c1_branch = paramsb
        v, x = data
        u_out_branch = fnn_B(v, W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch , c1_branch)
        u_out_branch = jnp.reshape(u_out_branch,(-1,G_dim,3))
        u_out_trunk = fnn_T(x, W_trunk, b_trunk,a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk) # predict on trunk
        u_out_trunk = jnp.einsum('jm,mi->ji',u_out_trunk,Rinv )
        u_pred = jnp.einsum('imn,jm->ijn',u_out_branch, u_out_trunk) # matmul

        return u_pred

    paramst=[]
    for i in range(nt):
        filename = foldert+modelt[i]
        temp = load_model(filename)
        paramst.append(temp)

    def choose_branch_model(folder,paramst,Rinv,data,u, tol):
        files = os.listdir(folder)
        f = []
        for i in range(nt):
            temp = []
            for file in files:
                if 'model'+str(i+1)+'.' in file:
                    temp.append(file)
            f.append(temp)
        model = []
        l2ensemble = []
        for i in range(nt):
            l2norm=[]
            for j in range(len(f[i])):
                filename = folder+f[i][j]
                print("filename=\t",filename)
                params = load_model(filename)
                pred = predict(params,paramst[i], Rinv[i], data, tol)
                l2 = jnp.mean(jnp.linalg.norm(u - pred, 2, axis=1)/np.linalg.norm(u , 2, axis=1))
                l2norm.append(l2)
            l2norm = jnp.array(l2norm)
            minimum = jnp.argmin(l2norm)
            model.append(f[i][minimum])
            l2ensemble.append(l2norm)
        return model,jnp.array(l2ensemble)
    folderb = './'+folderb+'/'
    filename = folderb+"Rmatrix"
    R = load_model(filename)

    Rinv = []
    for i in range(nt):
        temp = jnp.linalg.inv(R[i])
        Rinv.append(temp)

    modelb,l2norm = choose_branch_model(folderb,paramst,Rinv,[v_test,x_test],u_test, tol)
    print("modelb=\t",modelb)

    paramst = []
    for i in range(nt):
        filename = foldert+modelt[i]
        temp = load_model(filename)
        paramst.append(temp)

    paramsb = []
    for i in range(nt):
        filename = folderb+modelb[i]
        temp = load_model(filename)
        paramsb.append(temp)


    pred = []
    for i in range(nt):
        temp = predict(paramsb[i],paramst[i], Rinv[i] ,[v_train, x_train],tol)
        pred.append(temp)

    pred = jnp.array(pred)
    print("pred1=\t",pred.shape)

    pred_test = []
    for i in range(nt):
        temp = predict(paramsb[i],paramst[i], Rinv[i] ,[v_test, x_test],tol)
        pred_test.append(temp)

    pred_test = jnp.array(pred_test)

    if scaling=='01':
        pred = (pred)*(dmax-dmin)+dmin
        pred_test = (pred_test)*(dmax-dmin)+dmin
    else:
        pred = (pred+oness)*(dmax-dmin)/2.0+dmin
        pred_test = (pred_test+oness)*(dmax-dmin)/2.0+dmin

    if scaling=='01':
        u_train = (u_train)*(dmax-dmin)+dmin
        u_test = (u_test)*(dmax-dmin)+dmin
        v_train = (v_train)*(Xmax - Xmin)+ Xmin
        v_test = (v_test)*(Xmax - Xmin)+ Xmin
    else:
        u_train = (u_train+oness)*(dmax-dmin)/2.0+dmin
        u_test = (u_test+oness)*(dmax-dmin)/2.0+dmin
        v_train = (v_train+1.0)*(Xmax-Xmin)/2.0+Xmin
        v_test = (v_test+1.0)*(Xmax-Xmin)/2.0+Xmin

    if Problem== "HPR":
        u_train = u_train.at[:,:,0].set(jnp.exp(u_train[:,:,0]))
        u_train = u_train.at[:,:,2].set(jnp.exp(u_train[:,:,2]))

        u_test = u_test.at[:,:,0].set(jnp.exp(u_test[:,:,0]))
        u_test = u_test.at[:,:,2].set(jnp.exp(u_test[:,:,2]))

        pred = pred.at[:,:,:,0].set(jnp.exp(pred[:,:,:,0]))
        pred = pred.at[:,:,:,2].set(jnp.exp(pred[:,:,:,2]))

        pred_test = pred_test.at[:,:,:,0].set(jnp.exp(pred_test[:,:,:,0]))
        pred_test = pred_test.at[:,:,:,2].set(jnp.exp(pred_test[:,:,:,2]))

    l2_norm = []
    var = jnp.zeros((3,))
    for i in range(nt):
        err_test = jnp.mean(jnp.linalg.norm(u_test - pred_test[i,:,:,:], 2, axis=1)/\
                        np.linalg.norm(u_test , 2, axis=1),axis=0)
        var = err_test
        l2_norm.append(var)

    l2_norm = np.array(l2_norm)
    model = np.array(range(nt))
    print("l2_norm=\t",l2_norm.shape)
    l2_norm_mean = np.mean(l2_norm,axis=0)
    l2_norm_std = np.std(l2_norm,axis=0)

    print("L2 norm variables of the test dataset=\t",l2_norm_mean)
    print("std     variables of the test dataset=\t",l2_norm_std)

    print("L2 norm total of the test dataset=\t",np.mean(l2_norm_mean))
    print("std     total of the test dataset=\t",np.mean(l2_norm_std))

    pred_test_std= jnp.std(pred_test,axis=0)
    pred_test= jnp.mean(pred_test,axis=0)

    pred_std = jnp.std(pred,axis=0)
    pred = jnp.mean(pred,axis=0)

    filename = folderb+'loss'
    with open(filename, 'rb') as file:
        loss_datab = pickle.load(file)

    epo = loss_datab[0]

    loss_train = loss_datab[1]

    # Processing the loss function
    epo = np.array(epo)
    loss = np.zeros((epo.shape[0],nt))
    for i in range(epo.shape[0]):
        loss[i,:] = loss_train[i*(nt):(i+1)*(nt)]

    loss_std = np.std(loss,axis=-1)
    loss = np.mean(loss,axis=-1)

    filename = foldert+'loss'
    with open(filename, 'rb') as file:
        loss_datab = pickle.load(file)

    epo = loss_datab[0]

    loss_train = loss_datab[1]

    # Processing the loss function
    epo = np.array(epo)
    loss_T = np.zeros((epo.shape[0],nt))
    for i in range(epo.shape[0]):
        loss_T[i,:] = loss_train[i*(nt):(i+1)*(nt)]

    loss_std_T = np.std(loss_T,axis=-1)
    loss_T = np.mean(loss_T,axis=-1)


    err = np.abs(pred-u_train)
    err_test= np.abs(pred_test-u_test)

    Result_folder = './'+Result_folder+'/'

    def plotdata(x,data_p,data_e,std,xtitle,ytitle,label,filename,tag):
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(17)
        fig.set_figheight(10)
        m = len(data_e)
        # print(m)
        fig ,ax  = plt.subplots(1,1)
        if tag == 2:
            for i in range(m):
                ax.plot(x,data_e[i,:],label[i][2],label=label[i][0])
                ax.plot(x,data_p[i,:],label[i][3],label=label[i][1])
        elif tag == 21:
            for i in range(m):
                ax.plot(x,data_e[i,:],label[i][2],label=label[i][0])
                ax.plot(x,data_p[i,:],label[i][3],label=label[i][1])
            if Problem == 'HPR':
                ax.set_yscale("log")
                
        elif tag==3:
            m=1
            for i in range(m):
                ax.plot(x,data_e[:,i],label[i][2],label=label[i][0])
        elif tag==1:
            for i in range(m):
                ax.plot(x,data_e[i,:],label[i][2],label=label[i][0])
                ax.set_yscale("log")
        elif tag==5:
            for i in range(m):
                ax.plot(x,data_e[i,:],label=label[i])
                ax.set_ylim(bottom=0.0, top=2.0)

        elif tag==6:
            for i in range(m):
                ax.plot(x,data_e[i,:],label=label[i])
        else:
            for i in range(m):
                ax.plot(x,data_e[i,:],label[i][2],label=label[i][0])
                ax.set_yscale("log")

        ax.legend(loc='best')
        ax.set_xlabel(xtitle,fontsize=16)
        ax.set_ylabel(ytitle,fontsize=16)
        plt.savefig(filename,dpi=300)
        plt.close()

    data_p = pred[0:-1:100,:,0]
    data_e = u_train[0:-1:100,:,0]
    data_std = pred_std[0:-1:100,:,0]
    label_dat=v_train[0:-1:100,0]
    # print(data_p.shape)
    label=[]
    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[0]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[0]))+")")
    temp.append("-k")
    temp.append("--b")
    label.append(temp)

    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[1]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[1]))+")")
    temp.append("-k")
    temp.append("--r")
    label.append(temp)

    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[2]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[2]))+")")
    temp.append("-k")
    temp.append("--g")
    label.append(temp)

    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[3]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[3]))+")")
    temp.append("-k")
    temp.append("--m")

    label.append(temp)

    xtitle = r"$x$"
    ytitle = r"$\rho$"
    # print(x)
    filename = Result_folder+"rho_train"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,21)

    data_p = pred[0:-1:100,:,1]
    data_e = u_train[0:-1:100,:,1]
    data_std = pred_std[0:-1:100,:,1]

    xtitle = r"$x$"
    ytitle = r"$u$"
    # print(x)
    filename = Result_folder+"u_train"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,2)

    data_p = pred[0:-1:100,:,2]
    data_e = u_train[0:-1:100,:,2]
    data_std = pred_std[0:-1:100,:,2]

    xtitle = r"$x$"
    ytitle = r"$p$"
    # print(x)
    filename = Result_folder+"p_train"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,21)

    label=[]
    temp = []
    temp.append("Error(pl = "+str(label_dat[0])+")")
    temp.append("exac(pl = "+str(label_dat[0])+")")
    temp.append("-b")
    label.append(temp)

    temp = []
    temp.append("Error(pl = "+str(label_dat[1])+")")
    temp.append("exac(pl = "+str(label_dat[1])+")")
    temp.append("-r")
    label.append(temp)

    temp = []
    temp.append("Error(pl = "+str(label_dat[2])+")")
    temp.append("exac(pl = "+str(label_dat[2])+")")
    temp.append("-g")
    label.append(temp)

    temp = []
    temp.append("Error(pl = "+str(label_dat[3])+")")
    temp.append("exac(pl = "+str(label_dat[3])+")")
    temp.append("-m")
    label.append(temp)

    data_p = err[0:-1:100,:,2]
    data_e = err[0:-1:100,:,2]
    data_std = pred_std[0:-1:100,:,2]

    xtitle = r"$x$"
    ytitle = r"Absolute error $(p)$"
    # print(x)
    filename = Result_folder+"perorr_train"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,1)

    data_p = err[0:-1:100,:,1]
    data_e = err[0:-1:100,:,1]
    data_std = pred_std[0:-1:100,:,1]

    xtitle = r"$x$"
    ytitle = r"Absolute error $(u)$"
    # print(x)
    filename = Result_folder+"uerorr_train"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,1)

    data_p = err[0:-1:100,:,0]
    data_e = err[0:-1:100,:,0]
    data_std = pred_std[0:-1:100,:,0]

    xtitle = r"$x$"
    ytitle = r"Absolute error $(\rho)$"
    # print(x)
    filename = Result_folder+"rhoerorr_train"+".png"

    plotdata(x_train,data_p,data_e,pred_std,xtitle,ytitle,label,filename,1)

    # Plot test data
    data_p = pred_test[0:-1:25,:,0]
    data_e = u_test[0:-1:25,:,0]
    data_std = pred_test_std[0:-1:25,:,0]
    label_dat=v_test[0:-1:25,0]

    label=[]
    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[0]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[0]))+")")
    temp.append("-k")
    temp.append("--b")

    label.append(temp)

    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[1]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[1]))+")")
    temp.append("-k")
    temp.append("--r")
    label.append(temp)

    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[2]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[2]))+")")
    temp.append("-k")
    temp.append("--g")
    label.append(temp)

    temp = []
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[3]))+")")
    temp.append("pred(pl = "+str('{:.3e}'.format(label_dat[3]))+")")
    temp.append("-k")
    temp.append("--m")
    label.append(temp)

    xtitle = r"$x$"
    ytitle = r"$\rho$"
    # print(x)
    filename = Result_folder+"rho_test"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,21)

    data_p = pred_test[0:-1:25,:,1]
    data_e = u_test[0:-1:25,:,1]
    data_std = pred_test_std[0:-1:25,:,1]

    xtitle = r"$x$"
    ytitle = r"$u$"
    # print(x)
    filename = Result_folder+"u_test"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,2)

    data_p = pred_test[0:-1:25,:,2]
    data_e = u_test[0:-1:25,:,2]
    data_std = pred_test_std[0:-1:25,:,2]

    xtitle = r"$x$"
    ytitle = r"$p$"

    filename = Result_folder+"p_test"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,21)

    # Error Ploting

    label=[]
    temp = []
    temp.append("Error(pl = "+str('{:.3e}'.format(label_dat[0]))+")")
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[0]))+")")
    temp.append("-b")
    label.append(temp)

    temp = []
    temp.append("Error(pl = "+str('{:.3e}'.format(label_dat[1]))+")")
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[1]))+")")
    temp.append("-r")
    label.append(temp)

    temp = []
    temp.append("Error(pl = "+str('{:.3e}'.format(label_dat[2]))+")")
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[2]))+")")
    temp.append("-g")
    label.append(temp)

    temp = []
    temp.append("Error(pl = "+str('{:.3e}'.format(label_dat[3]))+")")
    temp.append("exac(pl = "+str('{:.3e}'.format(label_dat[3]))+")")
    temp.append("-m")
    label.append(temp)

    data_p = err_test[0:-1:25,:,2]
    data_e = err_test[0:-1:25,:,2]
    data_std = pred_test_std[0:-1:25,:,2]

    xtitle = r"$x$"
    ytitle = r"Absolute error $(p)$"
    # print(x)
    filename = Result_folder+"perorr_test"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,1)

    data_p = err_test[0:-1:25,:,1]
    data_e = err_test[0:-1:25,:,1]
    data_std = pred_test_std[0:-1:25,:,1]

    xtitle = r"$x$"
    ytitle = r"Absolute error $(u)$"
    # print(x)
    filename = Result_folder+"uerorr_test"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,1)

    data_p = err_test[0:-1:25,:,0]
    data_e = err_test[0:-1:25,:,0]
    data_std = pred_test_std[0:-1:25,:,1]

    xtitle = r"$x$"
    ytitle = r"Absolute error $(\rho)$"
    # print(x)
    filename = Result_folder+"rhoerorr_test"+".png"

    plotdata(x_train,data_p,data_e,data_std,xtitle,ytitle,label,filename,1)

    fig, ax = plt.subplots(1,1)
    fig.set_figwidth(17)
    fig.set_figheight(10)
    ax.plot(model,l2_norm[:,0],'ob',label='Density')
    ax.plot(model,l2_norm[:,1],'or',label='Velocity')
    ax.plot(model,l2_norm[:,2],'og',label='Pressure')
    ax.set_yscale("log")
    ax.legend(loc='best')
    filename = Result_folder+'accuracy.png'
    ax.set_xlabel('model',fontsize=16)
    ax.set_ylabel('L2',fontsize=16)
    plt.savefig(filename)
    plt.close()

    model = jnp.array(range(l2norm.shape[-1]))
    for i in range(nt):
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(17)
        fig.set_figheight(10)
        ax.plot(model,l2norm[i,:],'ob',label='Density')
        ax.set_yscale("log")
        ax.legend(loc='best')
        filename = Result_folder+'l2'+str(i)+'.png'
        ax.set_xlabel('model',fontsize=16)
        ax.set_ylabel('L2',fontsize=16)
        plt.savefig(filename)
        plt.close()

    label=[]
    temp = []
    temp.append("Train Loss")
    temp.append("Stand. Dev.")
    temp.append("-b")
    label.append(temp)

    data_p = loss
    data_e = loss
    data_std = loss_std

    data_p = data_p[None,:]
    data_e = data_e[None,:]
    data_std = data_std[None,:]

    xtitle = "Epoch"
    ytitle = "Loss_train_branch"
    # print(x)
    filename = Result_folder+"Loss_train_branch"+".png"

    plotdata(epo,data_p,data_e,data_std,xtitle,ytitle,label,filename,4)

    label=[]
    temp = []
    temp.append("Train Loss")
    temp.append("Stand. Dev.")
    temp.append("-b")
    label.append(temp)

    data_p = loss_T
    data_e = loss_T
    data_std = loss_std_T

    data_p = data_p[None,:]
    data_e = data_e[None,:]
    data_std = data_std[None,:]

    xtitle = "Epoch"
    ytitle = "Loss_train_trunk"
    # print(x)
    filename = Result_folder+"Loss_train_trunk"+".png"

    plotdata(epo,data_p,data_e,data_std,xtitle,ytitle,label,filename,4)
