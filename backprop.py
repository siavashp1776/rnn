import numpy as np

def hgrad(h,n,Why,scores,Whh):
    a = np.matmul(np.transpose(Why.val),scores.grad[:,n-1])
    b = h.grad[:,n]*(1-np.power(h.val[:,n],2))
    c = np.matmul(np.transpose(Whh.val),b)
    h.grad[:,n-1] = a+c
    return h

def cgrad(c,h,n):
    c.grad[:,n-1]= h.grad[:,n]*(1-np.power(h.val[:,n],2))
    return c

def scoregrad(scores,x,n,softmax):
    scores.grad[:,n-1]= ((-1*x.val[:,n-1])+1)*softmax.val[:,n-1]+x.val[:,n-1]*(softmax.val[:,n-1]-1)
    return scores

def Whygrad(Why,h,n,scores):
    Why.grad += np.transpose(np.tensordot(h.val[:,n-1],np.transpose(scores.grad[:,n-1]),0))
    return Why

def Whhgrad(Whh,h,c,n):
    Whh.grad += np.transpose(np.tensordot(h.val[:,n-1],np.transpose(c.grad[:,n-1]),0))
    return Whh

def Wxhgrad(Wxh,x,c,n):
    Wxh.grad += np.transpose(np.tensordot(x.val[:,n-1],np.transpose(c.grad[:,n-1]),0))
    return Wxh

def htgrad(h,Why,scores,t): ### for last hidden state in RNN (h at time t)
    h.grad[:,t-1] = np.matmul(np.transpose(Why.val),scores.grad[:,t-1])
    return h