import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.tree_util import tree_map
from jax.nn import relu, softmax, sigmoid, log_softmax
from jax.scipy.special import logsumexp
from functools import partial

import pickle

'''
This code is highly inspired by the tutorial "Get Started with JAX" by Aleksa Gordić.
Refer to Gordić, Aleksa. "Get started with JAX." GitHub repository, 2021.
Available at: https://github.com/gordicaleksa/get-started-with-JAX
'''

def MLP_init(layers : list) -> None:
    params = []
    seed = 0 
    parent_key = random.PRNGKey(seed)

    keys = random.split(parent_key, len(layers)-1)
    scale = 0.01 # reduce the initial variance of weigths
    for n_in,n_out,key in zip(layers[:-1],layers[1:],keys):
        params.append({
            'W': scale*random.normal(key, (n_in,n_out)),
            'b': scale*jnp.zeros(n_out)
        })

    return params

@jit
def MLP_predict(params, x):
    hidden_layers = params[:-1]

    act = x
    for layer in hidden_layers:
        act = jnp.dot(act, layer['W']) + layer['b']
        act = relu(act)
    
    w_last, b_last = params[-1]['W'], params[-1]['b']
    act = jnp.dot(act, w_last) + b_last
    return act - logsumexp(act)

batched_MLP_predict = vmap(MLP_predict, in_axes=(None, 0))

def loss_fn(params, imgs, labels):
    pred = batched_MLP_predict(params, imgs)
    return -jnp.mean(labels*pred)

def accuracy(params, imgs, labels):
    pred = batched_MLP_predict(params, imgs)
    return jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(labels, axis=1))

@jit
def update(params, imgs, labels, lr=1e-3):
    loss,grads = value_and_grad(loss_fn)(params, imgs, labels)
    return loss, tree_map(lambda p,g: p - lr*g, params, grads)

def train(params, imgs, labels, epochs = 100, lr=1e-3):
    for epoch in range(epochs):
        loss, grads = value_and_grad(loss_fn)(params, imgs, labels)
        params = tree_map(lambda p,g: p - lr*g, params, grads)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss {loss}, accuracy {accuracy(params, imgs, labels)}")

def save_model(params, path):
    with open(path, 'wb') as f:
        pickle.dump(params, f)