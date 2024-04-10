import jax
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import optax
import pickle
import jaxopt
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.example_libraries import optimizers

def train_branchnet(inputs):
    num_epochs = inputs['Epochs']['Branch']+1
    lr = inputs['learning_rate']
    nt = inputs['Ensembles']
    scaling = inputs['Data_scaling']
    foldert = inputs['Model_folder']['Trunk']
    folder = inputs['Model_folder']['Branch']
    Problem = inputs['Problem']
    FSM = inputs['FSM']['Branch']
    FRL = inputs['FRL']['Branch']
    layers_f = inputs['Architecture']['Branch']
    activation = inputs['RBA']['Branch']
    BasefuncB = getattr(jnp, activation)
    activation = inputs['RBA']['Trunk']
    BasefuncT = getattr(jnp, activation)
    decomposition = inputs['Decomposition']

    command = 'rm -r '+folder
    os.system(command)
    command = 'mkdir '+folder
    os.system(command)

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


    def save_model(param,n,k):
        filename = './'+folder+'/model'+str(k)+'.'+str(n)
        with open(filename, 'wb') as file:
            pickle.dump(param, file)

    def load_model(filename):
        with open(filename, 'rb') as file:
            param=pickle.load(file)
        return param

    initializer = jax.nn.initializers.glorot_normal()

    def hyper_initial_WB(layers,key):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2.0/(in_dim+out_dim))
            weight = initializer(key, (in_dim, out_dim), jnp.float32)*std
            bias = initializer(key, (1, out_dim), jnp.float32)*std
            W.append(weight)
            b.append(bias)
        return W, b

    def hyper_parameters_A(shape):
        return jnp.full(shape, 0.1, dtype=jnp.float32)

    def hyper_parameters_amplitude(shape):
        return jnp.full(shape, 0.0, dtype=jnp.float32)

    def hyper_parameters_freq1(shape):
        return jnp.full(shape, 0.1, dtype=jnp.float32)

     

    def hyper_initial_frequencies(layers):

        L = len(layers)

        a = []
        c = []

        a1 = []
        F1 = []
        c1 = []
        
        for l in range(1, L):

            a.append(hyper_parameters_A([1]))
            c.append(hyper_parameters_A([1]))

            a1.append(hyper_parameters_amplitude([1]))
            F1.append(hyper_parameters_freq1([1]))
            c1.append(hyper_parameters_amplitude([1]))

        return a, c, a1, F1, c1 

    def fnn_B(X, W, b, a, c, a1, F1, c1):
        inputs = X#2.*(X - Xmin)/(Xmax - Xmin) - 1.0
        # print("first input=\t",inputs.shape)
        L = len(W)
        for i in range(L-1):
            inputs =  BasefuncB(jnp.add(10*a[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c[i])) \
                + 10*a1[i]*jnp.sin(jnp.add(10*F1[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c1[i])) 
        Y = jnp.dot(inputs, W[-1]) + b[-1]  
        # print("Y=\t",Y.shape)   
        return Y

    def fnn_T(X, W, b, a, c, a1, F1, c1):
        inputs = X#2.*(X - Xmin)/(Xmax - Xmin) - 1.0
        # print("T first input=\t",inputs.shape)
        L = len(W)
        for i in range(L-1):
            inputs =  BasefuncT(jnp.add(10*a[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c[i])) \
                + 10*a1[i]*jnp.sin(jnp.add(10*F1[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c1[i])) 
        Y = jnp.dot(inputs, W[-1]) + b[-1]     
        return Y

    # #input dimension for Branch Net
    # u_dim = 1

    #output dimension for Branch and Trunk Net
    G_dim = int(np.ceil(layers_f[-1]/3))

    #Branch Net
    # layers_f = [u_dim] + [150]*5 + [3*G_dim]
    print("branch layers:\t",layers_f)

    key = random.PRNGKey(1234)
    keys = random.split(key, num=nt)
    # keym = random.PRNGKey(4234)
    # keysm = random.split(keym, num=nt)
    print("keysss=\t",keys)
    a_branch, c_branch, a1_branch, F1_branch , c1_branch = hyper_initial_frequencies(layers_f)
    W_branch, b_branch = [], []

    for i in range(nt):
        w1, b1 = hyper_initial_WB(layers_f,keys[i])
        W_branch.append(w1)
        b_branch.append(b1)
            

    def predict1(params, data):
        Am, W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk = params
        v, x = data
        Am = jnp.reshape(Am,(-1,G_dim,3))
        u_out_trunk = fnn_T(x, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk)
        return u_out_trunk,Am

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
        # print("f=\t",f)
        model = []
        for i in range(nt):
            # paramst=[]
            l2norm=[]
            # print("len(f[i])=\t",len(f[i]))
            for j in range(len(f[i])):
                filename = foldert+f[i][j]
                # print("filename=\t",filename)
                params = load_model(filename)
                # print("len=\t",len(params))
                pred = predictT(params, data)
                l2 = jnp.mean(jnp.linalg.norm(u - pred, 2, axis=1)/np.linalg.norm(u , 2, axis=1))
                l2norm.append(l2)
            l2norm = jnp.array(l2norm)
            minimum = jnp.argmin(l2norm)
            model.append(f[i][minimum])
            # print("l2norm=\t",l2norm)
            # print("l2norm=\t",minimum)
        return model
    foldert = './'+foldert+'/'
    modelt = choose_trunk_model(foldert,[v_train, x_train],u_train, tol)
    print("modelt=\t",modelt)

    i=0
    paramst = []
    for i in range(nt):
        filename = foldert+modelt[i]
        param = load_model(filename)
        paramst.append(param)

    u_train1 = []
    Rm = []
    for i in range(nt):
        phi , Am = predict1(paramst[i],[v_train, x_train])
        if decomposition == "SVD":
            Q,Sd,Vd = jnp.linalg.svd(phi, full_matrices=False)
            R = jnp.matmul(jnp.diag(Sd),Vd)
        else:
            Q,R = jnp.linalg.qr(phi, mode='reduced')

        Rm.append(R)
        temp = jnp.einsum('ij,kjm->kim',R, Am)
        print("phi=\t",phi.shape,i)
        print("Am=\t",Am.shape,i)
        print("Q=\t",Q.shape,i)
        print("R=\t",R.shape,i)
        print("u_train1=\t",temp.shape,i)
        u_train1.append(temp)

    def predict(params, data, tol):
        W_branch,b_branch,a_branch, c_branch, a1_branch, F1_branch , c1_branch = params
        v, x = data
        u_out_branch = fnn_B(v, W_branch, b_branch,a_branch, c_branch, a1_branch, F1_branch , c1_branch)
        # print("u_out_branch=\t",u_out_branch.shape)
        u_pred = jnp.reshape(u_out_branch,(-1,G_dim,3))
        # print("u_pred=\t",u_pred.shape)
        return u_pred


    def loss(params, data, u , tol):
        u_preds = predict(params, data ,tol)
        loss_data = jnp.mean((u_preds.flatten() - u.flatten())**2)
        mse = loss_data 
        return mse

    @jit
    def update(params, data, u, opt_state, tol):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = value_and_grad(loss,argnums=0,has_aux=False)(params, data, u, tol)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value

    opt_init, opt_update, get_params = optimizers.adam(lr)


    opt_state = []
    for i in range(nt):
        opt_state.append(opt_init([W_branch[i], b_branch[i], a_branch, c_branch, a1_branch, F1_branch , c1_branch]))

    params = []
    for i in range(nt):
        params.append(get_params(opt_state[i]))



    train_loss, test_loss = [], []

    epo = []
    start_time = time.time()
    start_time1= time.time()
    n = 0

    for epoch in range(num_epochs):
        for i in range(nt):
            params[i], opt_state[i], loss_val = update(params[i], [v_train, x_train], u_train1[i], opt_state[i], tol)
        if epoch % FSM == 0:
            for i in range(nt):
                save_model(params[i],n,i+1)
            n += 1
        if epoch % FRL ==0:
            for i in range(nt):
                epoch_time = time.time() - start_time
                u_train_pred = predict(params[i], [v_train, x_train], tol)
                err_train = jnp.mean(jnp.linalg.norm(u_train1[i] - u_train_pred, 2, axis=1)/\
                    np.linalg.norm(u_train1[i] , 2, axis=1))
                l1 = loss(params[i], [v_train, x_train], u_train1[i], tol)
                train_loss.append(l1) 
                print("Epoch {} | T: {:0.6f} | Branch {} | Train MSE: {:0.3e} | Train L2: {:0.6f}".format(epoch, epoch_time, i,\
                                                                    l1, err_train))
            epo.append(epoch)

        start_time = time.time()

    total_time = time.time() - start_time1
    print("training time for the branch net=\t",total_time)
    for i in range(nt):
        save_model(params[i],n,i+1)

    filename = './'+folder+'/Rmatrix'
    with open(filename, 'wb') as file:
        pickle.dump(Rm, file)

    filename = './'+folder+'/loss'
    with open(filename, 'wb') as file:
        pickle.dump((epo,train_loss), file)

