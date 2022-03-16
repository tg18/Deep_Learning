import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

#plt.rcParams['figure.figsize']=[16,8]
#fig,((ax1,ax2,ax3,ax4,ax5,ax6))=plt.subplots(6,1)
fig, axs = plt.subplots(1, 7, figsize=(30, 5))
#plt.subplot((6,1))
picture=imread("pic.jpg")
X=np.mean(picture,-1)
U,S,V=np.linalg.svd(X,full_matrices=False)
S=np.diag(S)
j=0

for r in (5,10,15,20,50,100):
    img_approx = U[:,:r] @ S[0:r,:r] @ V[:r,:]
    #plt.figure(j+1)
    j=j+1
        #plt.subplot(6,1,j)
    axs[j].plot(img_approx)
    #axs[j].set_cmap('viridis') #('gray')
    plt.axis('on')
    plt.title('r = '+str(r))

    #axs[j].plot()
plt.show()
    #plt.subplot(6, 1, j)

