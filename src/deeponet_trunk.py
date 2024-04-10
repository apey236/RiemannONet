import jax
import os
import numpy as np
import time
import scipy.io as io
import jax.numpy as jnp
import pickle
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.example_libraries import optimizers

def train_trunknet(inputs):
    num_epochs = inputs['Epochs']['Trunk']+1
    lr = inputs['learning_rate']
    nt = inputs['Ensembles']
    scaling = inputs['Data_scaling']
    folder = inputs['Model_folder']['Trunk']
    Problem = inputs['Problem']
    FSM = inputs['FSM']['Trunk']
    FRL = inputs['FRL']['Trunk']
    layers_x = inputs['Architecture']['Trunk']
    activation = inputs['RBA']['Trunk']
    Basefunc = getattr(jnp, activation)
    # print(activation)
    # exec(f'baseFunc = jnp.{activation}') 

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

    print("v_train",v_train.shape)
    print("x_train",x_train.shape)
    print("u_train",u_train.shape)
    print("v_test",v_test.shape)
    print("x_test",x_test.shape)
    print("u_test",u_test.shape)


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

    def matrix_init(N,K,key):
        in_dim = N
        out_dim = K
        std = np.sqrt(2.0/(in_dim+out_dim))
        W = initializer(key, (in_dim, out_dim), jnp.float32)*std
        return W

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

    def fnn_T(X, W, b, a, c, a1, F1, c1):
        inputs = X#2.*(X - Xmin)/(Xmax - Xmin) - 1.0
        # print("T first input=\t",inputs.shape)
        L = len(W)
        for i in range(L-1):
            inputs = Basefunc(jnp.add(10*a[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c[i])) \
                + 10*a1[i]*jnp.sin(jnp.add(10*F1[i]*jnp.add(jnp.dot(inputs, W[i]), b[i]),c1[i])) 
        Y = jnp.dot(inputs, W[-1]) + b[-1]     
        return Y

    # # Trunk dim
    # x_dim = 1

    #output dimension for Branch and Trunk Net
    G_dim = layers_x[-1]

    #Trunk Net
    # layers_x = [x_dim] + [150]*5 + [G_dim]

    key = random.PRNGKey(1234)
    keys = random.split(key, num=nt)
    keym = random.PRNGKey(4234)
    keysm = random.split(keym, num=nt)
    print("keysss=\t",keys)

    a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk = hyper_initial_frequencies(layers_x)

    W_trunk, b_trunk = [], []
    Am = []
    for i in range(nt):
        w1, b1 = hyper_initial_WB(layers_x,keys[i])
        A = matrix_init(3*G_dim,u_train.shape[0],keysm[i])
        W_trunk.append(w1)
        b_trunk.append(b1)
        Am.append(A)


    def predict(params, data, tol):
        Am, W_trunk, b_trunk,a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk = params
        v, x = data
        Am = jnp.reshape(Am,(-1,G_dim,3))
        u_out_trunk = fnn_T(x, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk , c1_trunk)
        u_pred = jnp.einsum('imn,jm->ijn',Am, u_out_trunk) # matmul

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
        opt_state.append(opt_init([Am[i], W_trunk[i], b_trunk[i],a_trunk, c_trunk,a1_trunk, F1_trunk, c1_trunk]))

    params = []
    for i in range(nt):
        params.append(get_params(opt_state[i]))

    train_loss, test_loss = [], []

    epo = []
    start_time = time.time()
    start_time1 = time.time()
    n = 0

    for epoch in range(num_epochs):
        
        for i in range(nt):
            params[i], opt_state[i], loss_val = update(params[i], [v_train, x_train], u_train, opt_state[i], tol)
        
        if epoch % FSM == 0:
            for i in range(nt):
                save_model(params[i],n,i+1)
            n += 1

        if epoch % FRL ==0:
            for i in range(nt):
                epoch_time = time.time() - start_time
                u_train_pred = predict(params[i], [v_train, x_train], tol)
                err_train = jnp.mean(jnp.linalg.norm(u_train - u_train_pred, 2, axis=1)/\
                    np.linalg.norm(u_train , 2, axis=1))
                l1 = loss(params[i], [v_train, x_train], u_train, tol)
                train_loss.append(l1) 
                print("Epoch {} | T: {:0.6f} | Trunk {} | Train MSE: {:0.3e} | Train L2: {:0.6f}".format(epoch, epoch_time, i,\
                                                                    l1, err_train))
            epo.append(epoch)

        start_time = time.time()

    total_time = time.time() - start_time1
    print("training time for the trunk net=\t",total_time)

    for i in range(nt):
        save_model(params[i],n,i+1)

    filename = './'+folder+'/loss'
    with open(filename, 'wb') as file:
        pickle.dump((epo,train_loss), file)

    pred = predict(params[0], [v_train, x_train], tol)
    if scaling=='01':
        pred = (pred)*(dmax-dmin)+dmin
    else:
        pred = (pred+oness)*(dmax-dmin)/2.0+dmin

    for i in range(1,nt):  
        pred1 = predict(params[i], [v_train, x_train], tol)
        if scaling=='01':
            pred1 = (pred1)*(dmax-dmin)+dmin
        else:
            pred1 = (pred1+oness)*(dmax-dmin)/2.0+dmin
        pred += pred1
    pred = pred/float(nt)

    pred1 = predict(params[0], [v_train, x_train], tol)
    if scaling=='01':
        pred1 = (pred1)*(dmax-dmin)+dmin
    else:
        pred1 = (pred1+oness)*(dmax-dmin)/2.0+dmin
    std = (pred1-pred)**2.0
    for i in range(1,nt): 
        pred1 = predict(params[i], [v_train, x_train], tol)
        if scaling=='01':
            pred1 = (pred1)*(dmax-dmin)+dmin
        else:
            pred1 = (pred1+oness)*(dmax-dmin)/2.0+dmin
        std += (pred1-pred)**2.0

    std = np.sqrt(std/float(nt))

    if scaling=='01':
        u_train = (u_train)*(dmax-dmin)+dmin
        u_test = (u_test)*(dmax-dmin)+dmin
    else:
        u_train = (u_train+oness)*(dmax-dmin)/2.0+dmin
        u_test = (u_test+oness)*(dmax-dmin)/2.0+dmin

