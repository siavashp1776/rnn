# RNN general structure:
#
# h_t = tanh(w_hh*h_t_1+w_xh*x_t)
# yt = softmax(w_hy*h_t)

import numpy as np
import backprop as b

# open file
# determine character / array mapping
# create 2D array for x, n letters * t depth
# hidden state size: 2-4X number of features. I say there are 100 features in the english language
# 400 size hidden state
# W_hh = 400 * 400

def read_y(y,ldict):
    pos = np.argmax(y,0)
    string = []
    for i in pos:
        string.append(ldict[i])
    phrase = ''.join(string)
    print(phrase)
    return phrase


def initialize_vars():

    listOfChars = list()
    with open("data.txt", "r") as myfile:
        for line in myfile:
            words = line.strip()
            for i in list(words):
                listOfChars.append(i)

    myset = set(listOfChars)
    n = len(myset)
    ldict = dict()
    lldict = dict()
    j = 0
    for i in myset:
        ldict.update({i: j})
        lldict.update({j:i})
        j += 1

    listOfIndxs = list()
    for a in listOfChars:
        listOfIndxs.append(ldict[a])

    t = len(listOfIndxs)

    xx = np.zeros((n, t))
    for i in range(t):  # set indices of x np array to 1 where the letter appears
        j = listOfIndxs[i]
        xx[j, i] = 1

    # FORWARD PASS
    class var:
        val = []
        grad = []


    x = var()
    x.val = xx
    x.grad = np.zeros_like(x.val)

    hs = 400  # size of hidden state
    w_hh = var()
    w_hh.val = np.random.rand(hs, hs) / 1000
    w_hh.grad = np.zeros_like(w_hh.val)

    w_xh = var()
    w_xh.val = np.random.rand(hs, n) / 100
    w_xh.grad = np.zeros_like(w_xh.val)

    w_hy = var()
    w_hy.val = np.random.rand(n, hs)
    w_hy.grad = np.zeros_like(w_hy.val)

    h = var()
    h.val = np.zeros((400, t))
    h.grad = np.zeros_like(h.val)

    scores = var()
    scores.val = np.zeros_like(x.val)
    scores.grad = np.zeros_like(scores.val)

    y = var()
    y.val = np.zeros_like(x.val)
    y.grad = np.zeros_like(y.val)

    J = var()
    J.val = np.zeros(t)
    J.grad = np.ones(t)

    c = var()
    c.val = np.zeros_like(h.val)
    c.grad = np.zeros_like(c.val)
    vars = [x,w_hh,w_xh, w_hy,h,scores,y,J,c]
    return [vars,lldict]

def softmaxf(x):
    a = x - np.max(x)
    return np.exp(a) / np.sum(np.exp(a))

def forward(vars):
    [x, w_hh, w_xh, w_hy, h, scores, y, J, c] = vars
    t = max(np.shape(x.val))
    for T in range(t):
        c.val[:,T] = np.matmul(w_hh.val, h.val[:, T - 1]) + np.matmul(w_xh.val, x.val[:, T])
        h.val[:, T] = np.tanh(c.val[:,T])
        scores.val[:,T] = np.matmul(w_hy.val, h.val[:, T])
        y.val[:, T] = softmaxf(scores.val[:,T])
        J.val[T] = np.sum(x.val[:, T] * np.log(y.val[:, T]))

    Js = -(1 / t) * np.sum(x.val * np.log(y.val))
    vars = [x, w_hh, w_xh, w_hy, h, scores, y, J, c]
    Perplexity = 2 ** Js
    return [vars, Js, Perplexity]

## BACKWARDS PASS

def backwards(vars):
    [x, w_hh, w_xh, w_hy, h, scores, y, J, c] = vars
    t = max(np.shape(x.val))
    for n in reversed(range(t)):
        scores = b.scoregrad(scores,x,n,y)

    h = b.htgrad(h,w_hy,scores,t)

    for n in reversed(range(t)):
        h = b.hgrad(h,n,w_hy,scores,w_hh)

    for n in reversed(range(t)):
        c = b.cgrad(c,h,n)

    for n in reversed(range(t)):
        w_hh = b.Whhgrad(w_hh,h,c,n)
        w_hy = b.Whygrad(w_hy,h,n,scores)
        w_xh = b.Wxhgrad(w_xh,x,c,n)
    return [x, w_hh, w_xh, w_hy, h, scores, y, J, c]

print('ye')
