import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.tree_util import tree_map
from jax.nn import relu, softmax, sigmoid, log_softmax
from jax.scipy.special import logsumexp
from functools import partial

'''
This code is highly inspired by the tutorial "Get Started with JAX" by Aleksa Gordić.
Refer to Gordić, Aleksa. "Get started with JAX." GitHub repository, 2021.
Available at: https://github.com/gordicaleksa/get-started-with-JAX
'''

class MLP:
    def __init__(self,layers : list) -> None:
        self.__params = []
        self.__activations_fn = [None]*(len(layers)-1)
        self.__loss_fn = None
        self.__optimizer = None
        self.__metrics = []
        self.__seed = 0 
        self.__parent_key = random.PRNGKey(self.__seed)

        keys = random.split(self.__parent_key, len(layers)-1)
        scale = 0.01 # reduce the initial variance of weigths
        for n_in,n_out,key in zip(layers[:-1],layers[1:],keys):
            self.__params.append({
                'W': scale*random.normal(key, (n_in,n_out)),
                'b': scale*jnp.zeros(n_out)
            })

    @property
    def params(self) -> list:
        return self.__params

    def set_activations(self, activations : list) -> None:
        if len(activations) != len(self.__activations_fn):
            raise ValueError("Number of activations must match the number of layers")
        
        for i, activation in enumerate(activations):
            if activation == "relu":
                self.__activations_fn[i] = relu
            elif activation == "sigmoid":
                self.__activations_fn[i] = sigmoid
            elif activation == "softmax":
                self.__activations_fn[i] = softmax
            elif activation == "log_softmax":
                self.__activations_fn[i] = log_softmax
            elif activation == "sigmoid":
                self.__activations_fn[i] = sigmoid
            elif activation == "tanh":
                self.__activations_fn[i] = jnp.tanh
            elif activation == "identity":
                self.__activations_fn[i] = lambda x: x
            else:
                raise ValueError(f"Activation function {activation} not supported")
            
        # cast to tuple
        self.__activations_fn = tuple(self.__activations_fn)

    @jit
    def predict_jit(params, x):
        #if activations_fn[0] is None:
        #    raise Warning("Activations not set")

        act = x
        for i,layer in enumerate(params[:-1]):
            act = jnp.dot(act,layer['W']) + layer['b']
            act = relu(act)

        w_last, b_last = params[-1]['W'], params[-1]['b']
        act = jnp.dot(act,w_last) + b_last

        return act - logsumexp(act)

    def predict(self, params, x):
        batched_predict = vmap(self.predict_jit, in_axes=(None,0))
        return batched_predict(params, x)

    def loss_fn(self,params,imgs,labels):
        pred = self.predict(x = imgs, params=params)
        return -jnp.mean(pred * labels)
    
    # this method does not work
    def accuracy(self,imgs,labels):
        pred = self.predict_batch(imgs)
        return jnp.mean(jnp.argmax(pred,axis=1) == jnp.argmax(labels,axis=1))

    @staticmethod
    @partial(jit, static_argnames=['loss_fn'])
    def __update_jit(loss_fn, params, imgs, labels,lr=0.01):
        loss,grads = value_and_grad(loss_fn)(params,imgs,labels)
        return loss,tree_map(lambda p,g: p-lr*g, params, grads)
    
    def train(self, X_train, y_train, epochs=100, lr=0.01):
        params = self.__params

        for i in range(epochs):
            loss, params = self.__update_jit(self.loss_fn, params, X_train, y_train, lr)
            
            if i % 10 == 0:
                print(f"Epoch {i}, Loss: {loss}")
            
        self.__params = params

    def print(self) -> None:
        print(f"MLP with {len(self.__params)} layers")
        print("-"*30)
        print(tree_map(lambda x: x.shape, self.__params))

from ETL import ETL

if __name__=="__main__":
    mlp = MLP([784,128,128,10])
    
    etl = ETL("mnist.npz")
    X_train, X_test, y_train, y_test = etl.loadData(reduce=True)
    img = X_train[0]

    print(img.shape)
    pred = mlp.predict(mlp.params,img)
    print(pred.shape)