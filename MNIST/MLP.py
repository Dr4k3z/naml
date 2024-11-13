import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.tree_util import tree_map
from jax.nn import relu, softmax, sigmoid, log_softmax
from jax.scipy.special import logsumexp
from functools import partial
from ETL import ETL, NumpyDataLoader
import time
import pickle

'''
This code is highly inspired by the tutorial "Get Started with JAX" by Aleksa Gordić.
Refer to Gordić, Aleksa. "Get started with JAX." GitHub repository, 2021.
Available at: https://github.com/gordicaleksa/get-started-with-JAX
'''

class MLP:
    def __init__(self, layers : list, seed : int = 0):
        """
            MLP Constructor
            layers : list, contains the number of neurons in each layer
        """

        self.__params = []
        self.__parent_key = random.PRNGKey(seed)

        keys = random.split(self.__parent_key, len(layers)-1)
        scale = 0.1 # reduce the initial variance of weigths
        for n_in,n_out,key in zip(layers[:-1],layers[1:],keys):
            self.__params.append({
                'W': scale*random.normal(key, (n_in,n_out)),
                'b': scale*jnp.zeros(n_out)
            })

    @partial(jit, static_argnums=(0,))
    def __predict(self, x : jnp.array, params : list) -> jnp.array:
        """
            Forward pass of the MLP. Works only for single inputs, in fact
            it'll be wrapped by batched_predict in the next function. 
            x : jnp.array, input array
            params : list, containing the model parameters
        """
        hidden_layers = params[:-1]

        act = x
        for layer in hidden_layers:
            act = jnp.dot(act, layer['W']) + layer['b']
            act = relu(act)
        
        w_last, b_last = params[-1]['W'], params[-1]['b']
        act = jnp.dot(act, w_last) + b_last
        return act - logsumexp(act)

    def batched_predict(self, x : jnp.arange, params : list =None) -> jnp.array:
        """
            Batched version of self.__predict. It either use the current
            parameters or the ones passed as argument. This is key for 
            training, since paramaters are updated at each iteration.

        """
        params = params if params is not None else self.__params
        return vmap(self.__predict, in_axes=(0, None))(x, params)

    def loss_function(self, params : list, imgs : jnp.array, labels : jnp.array) -> jnp.array:
        """
            Loss function for the MLP. It computes the cross-entropy loss
            for a batch of data.
            params : list, model parameters
            imgs : jnp.array, batch of images
            labels : jnp.array, batch of labels
        """
        pred = self.batched_predict(imgs, params)
        return -jnp.mean(labels*pred)    

    @partial(jit, static_argnums=(0,))
    def _update(self, params : list, imgs : jnp.array, labels : jnp.array, lr : float) -> tuple:
        """
            Parameters update function. It takes as input as batch of data 
            and performs single step batch gradient descent.
            params : list, model parameters
            imgs : jnp.array, batch of images
            labels : jnp.array, batch of labels
            lr : float, learning rate
        """
        loss,grads = value_and_grad(self.loss_function)(params, imgs, labels)
        return loss, tree_map(lambda p,g: p - lr*g, params, grads)
    
    def train(self, train_loader : NumpyDataLoader, epochs : int = 100, lr : float = 1e-3) -> None:
        """
            Training method. Could I jit this?
            train_loader : NumpyDataLoader, custom loader object
            epochs : int, number of epochs
            lr : float, learning rate
        """
        for epoch in range(epochs):
            for cnt,(img,label) in enumerate(train_loader):
                loss, self.__params = self._update(self.__params, img, label, lr)
                if cnt % 100 == 0:
                    print(f"Epoch {epoch}, batch {cnt}, loss {loss}")

    @property
    def params(self):
        """
            Getter function for MLP parameters, return in pytree format
        """
        return tree_map(lambda x : x.shape, self.__params)
        
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__params, f)

if __name__=="__main__":
    etl = ETL("mnist.npz")
    X_train,y_train,X_test,y_test = etl.loadData(reduce=False)

    batch_size = 128
    train_loader = NumpyDataLoader(X_train, y_train, batch_size, shuffle=True, drop_last=True)

    mlp = MLP([784,128,128,10])
    
    mlp.train(train_loader, epochs=10)

    print("Training error: ", mlp.loss_function(mlp.params, X_train, y_train))

    mlp.save_model("mlp_model.pkl")