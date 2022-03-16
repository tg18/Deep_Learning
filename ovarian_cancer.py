import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

X=np.loadtxt(os.path.join('ovariancancer_obs.csv'),delimiter=',')
#X=np.genfromtxt(os.path.join('ovariancancer_obs.csv'), delimiter=';')[:,:-1]

f=open(os.path.join('ovariancancer_obs.csv'),'r')

grp=np.loadtxt('ovariancancer_grp.csv',dtype=str)

U,S,VT=np.linalg.svd(X,full_matrices=0)

fig1=plt.figure()
ax1=fig1.add_subplot(121)
ax1.semilogy(S,'-o',color='k')
ax2=fig1.add_subplot(122)
ax2.plot(np.cumsum(S)/np.sum(S),'-o',color='k')

#plt.show()
fig2=plt.figure()
ax=fig2.add_subplot(111,projection='3d')

for j in range(X.shape[0]):
    x = VT[0,:] @ X[j,:].T
    y = VT[1,:] @ X[j,:].T
    z = VT[2,:] @ X[j,:].T

    if grp[j]=='Cancer':
        ax.scatter(x,y,z,marker='x',color='r',s=50)
    else:
        ax.scatter(x, y, z, marker='o', color='b', s=50)
ax.view_init(25,20)
plt.show()

