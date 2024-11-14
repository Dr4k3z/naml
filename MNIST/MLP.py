import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.tree_util import tree_map
from jax.nn import relu, softmax, sigmoid, log_softmax
from jax.scipy.special import logsumexp
from functools import partial
from ETL import ETL, DataLoader
import time
import pickle
import matplotlib.pyplot as plt

'''
This code is highly inspired by the tutorial "Get Started with JAX" by Aleksa Gordić.
Refer to Gordić, Aleksa. "Get started with JAX." GitHub repository, 2021.
Available at: https://github.com/gordicaleksa/get-started-with-JAX
'''

# TODO: Xavier initialization

class MLP:
    """
        List of activation functions supported by this class.
        They are mapped to the actual functions, which will be
        later used in the forward pass and training
    """
    activation_fns = {
        'relu': relu,
        'sigmoid': sigmoid,
        'softmax': softmax,
        'log_softmax': log_softmax,
        'identity': lambda x: x
    }

    loss_fns = {
        'cross_entropy': lambda y_true, y_pred: -jnp.mean(y_true*y_pred), # is this the cross-entropy loss?
        'mse': lambda y_true, y_pred: jnp.mean((y_true-y_pred)**2),
        'log-likelihood' : lambda y_true,y_pred : -jnp.mean(jnp.sum(y_true * y_pred, axis=1))  # Cross-entropy loss
    }

    def __init__(self, layers : list, activations : list = None, loss : str = "cross_entropy", seed : int = 0):
        """
            MLP Constructor
            layers : list, contains the number of neurons in each layer
            activations : list, contains the activation functions for each layer
            seed : int, seed for random number generator
        """

        self.__params = []
        self.__activations = activations or ['relu'] * (len(layers) - 1) # if None, just use relu
        self.__loss = loss
        self.__parent_key = random.PRNGKey(seed)

        keys = random.split(self.__parent_key, len(layers)-1)
        for n_in,n_out,key in zip(layers[:-1],layers[1:],keys):
            scale = jnp.sqrt(2/n_in) # He initialization
        
            self.__params.append({
                'W': scale*random.normal(key, (n_in,n_out)),
                'b': scale*jnp.zeros(n_out)
            })

    def __get_activation(self,activation : str):
        """
            This method maps the string of activation functions name
            to the actual functions. Used in predict and training.
        """
        return self.activation_fns[activation]
    
    def __get_loss(self, loss : str):
        """
            This method maps the string of loss functions name
            to the actual function. Used in loss and training.
        """
        return self.loss_fns[loss]

    @classmethod
    def load_model(cls, path : str):
        """
            Load model parameters from pickel file. The method creates a new instance
            of the class, bypassing the constructor (how does this work is still obscure). 
            path : str, path to the pickel file
        """
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls.__new__(cls)
        model.__params = model_state['params']
        model.__activations = model_state['activations']
        model.__loss = model_state['loss']
        model.__lr = model_state['lr']
        model.__parent_key = model_state['parent_key']
        return model

    @partial(jit, static_argnums=(0,))
    def __predict(self, x : jnp.array, params : list) -> jnp.array:
        """
            Forward pass of the MLP. Works only for single inputs, in fact
            it'll be wrapped by batched_predict in the next function. 
            x : jnp.array, input array
            params : list, containing the model parameters
        """
        hidden_layers = params[:-1]
        hidden_activations = self.__activations[:-1]

        assert len(hidden_layers) == len(hidden_activations), "Number of layers and activations must match"

        act = x
        for layer,activation in zip(hidden_layers, hidden_activations):
            act = jnp.dot(act, layer['W']) + layer['b']
            act = self.__get_activation(activation)(act) # get's the actual func from the alias, and applies it
        
        w_last, b_last = params[-1]['W'], params[-1]['b']
        act = jnp.dot(act, w_last) + b_last
        return act - logsumexp(act) # this last line creates trouble
        #return self.__get_activation(self.__activations[-1])(act)

    def batched_predict(self, x : jnp.arange, params : list = None) -> jnp.array:
        """
            Batched version of self.__predict. It either use the current
            parameters or the ones passed as argument. This is key for 
            training, since paramaters are updated at each iteration.

        """
        params = params if params is not None else self.__params
        return vmap(self.__predict, in_axes=(0, None))(x, params)

    def loss_function(self, imgs : jnp.array, labels : jnp.array, params : list = None) -> jnp.array:
        """
            Loss function for the MLP. It computes the cross-entropy loss
            for a batch of data.
            params : list, model parameters
            imgs : jnp.array, batch of images
            labels : jnp.array, batch of labels
        """
        pred = self.batched_predict(imgs, params)
        return self.__get_loss(self.__loss)(labels, pred)
    
    @partial(jit, static_argnums=(0,))
    def __update(self, params : list, imgs : jnp.array, labels : jnp.array, lr : float) -> tuple:
        """
            Parameters update function. It takes as input as batch of data 
            and performs single step batch gradient descent.
            params : list, model parameters
            imgs : jnp.array, batch of images
            labels : jnp.array, batch of labels
            lr : float, learning rate
        """

        loss,grads = value_and_grad(self.loss_function, argnums=2)(imgs, labels, params)
        return loss, tree_map(lambda p,g: p - lr*g, params, grads)
    
    def train(self, train_loader : DataLoader, epochs : int = 100, lr : float = 1e-3, show : bool = True) -> None:
        """
            Training method. Could I jit this?
            train_loader : DataLoader, custom loader object
            epochs : int, number of epochs
            lr : float, learning rate
        """
        self.__lr = lr # save learning rate for later use
        for epoch in range(epochs):
            for cnt,(imgs,labels) in enumerate(train_loader):
                loss, self.__params = self.__update(self.__params, imgs, labels, lr)
                if show and cnt % 100 == 0:
                    print(f'Epoch {epoch}, batch {cnt}, loss {loss}')

    def save_model(self, path : str) -> None:
        """
            Save the model parameters, activation functions and hyperparameters
            in pickel file. We define a state datastructure that contains all
        """
        model_state = {
            'params': self.__params,
            'activations': self.__activations,
            'loss' : self.__loss,
            'lr' : self.__lr,
            'parent_key': self.__parent_key
        }

        with open(path, 'wb') as f:
            pickle.dump(model_state, f)

    @property
    def params(self):
        """
            Getter function for MLP parameters, return in pytree format
        """
        return self.__params
    
    @property
    def activations(self):
        """
            Getter function for activation functions
        """
        return self.__activations
    
    @property
    def loss(self):
        """
            Getter function for loss function
        """
        return self.__loss

if __name__=='__main__':
    etl = ETL('mnist.npz')
    X_train,y_train,X_test,y_test = etl.loadData(onehot=True)

    batch_size = 128
    train_loader = DataLoader(X_train, y_train, batch_size, shuffle=True, drop_last=True)
 
    #mlp = MLP([784,128,128,10])
    
    #mlp.train(train_loader, epochs=10, lr=1e-3, show=True)
    #mlp.save_model('mlp.pkl')
    #mlp = MLP.load_model("mlp.pkl")

    #print(f'Loss: {mlp.loss_function(X_train[:128], y_train[:128], mlp.params)}')

    mlp = MLP(
        [784, 128, 128, 10],
        activations = ['relu', 'relu', 'identity'],
        loss = 'cross_entropy'
    )
    
    mlp.train(train_loader, epochs=50, lr=1e-3, show=True)

    Y = mlp.batched_predict(X_train[:10])
    for i in range(10):
        print(jnp.argmax(Y[i]), y_train[i])    
