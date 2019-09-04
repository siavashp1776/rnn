import numpy as np
import sgd as SGD
import rnn as r
import matplotlib.pyplot as plt


## SET UP BATCH SGD PARAMETERS
alpha = 0.00001

iterations = 5001
acc = []
accArr = []
lossArr = []
parr = []

[vars,ldict] = r.initialize_vars()

## SELECT AND RUN BATCH
for iter in range(int(iterations)):
    [vars,Js,Perplexity] = SGD.select_and_run(vars)
    print('Loss:',Js)
    lossArr.append(Js)
    vars = SGD.weight_update(vars,float(alpha)/(0.05*int(iter+1)))
    print('Num examples: ', np.size(lossArr))
    print('')
    if iter%500 == 0:
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        x = [i for i in range(np.size(lossArr))]
        axes.plot(x, lossArr)
        fig.savefig('Loss.png')
        [x, w_hh, w_xh, w_hy, h, scores, y, J, c] = vars
        p = r.read_y(y.val, ldict)
        pp = '\n'+ str(iter)+':  '+p
        parr.append(pp)

with open('output.txt','a') as files:
    files.writelines(parr)
files.close()

print('done with all iterations')