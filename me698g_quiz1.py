import pandas as pd
import numpy as np

hs=[10,8,5,2] # hours studied
hp=[3,20,1,6] # hours playing
cm=[0,2,5,8] #class missed
pc=[5,7,2,1] #practice coding
grades=[87,75,63,41]
x=[hs,hp,cm,pc]
x=np.array((x))
print(np.transpose(x))

w=np.ones(4)

parameters=4
iteration=1000
neu=0.00005 #learning rate

w=np.ones(parameters,dtype=np.float64)
w.reshape((parameters,1))
grad_error=np.zeros(parameters,dtype=np.float64)

for n in range(iteration):
    for i in range(parameters):
        grad_error[i]=np.matmul(np.matmul(np.transpose(x),w)-grades,x[:,i])
    grad_error.reshape((parameters,1))
    w=w-neu*grad_error
print('w=',w)
x_test=[8,20,2,7]
y_pred=np.matmul(x_test,w)
print('grades of test student are=',y_pred)