import numpy as np
import rnn as r

def select_and_run(vars):
    [vars, Js, Perplexity] = r.forward(vars)
    vars = r.backwards(vars)
    return [vars, Js, Perplexity] ### stochastic with the last elem in batch

def weight_update(vars,alpha):
    for var in vars:
        var.val -= alpha* var.grad
    return vars
