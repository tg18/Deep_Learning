from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import glob


import cv2 as cv

path = glob.glob("JPGS/*.jpg")
IMG = []
for img in path:
    n = cv.imread(img)
    IMG.append(n)

IMG=np.array(IMG)
print(IMG.shape)

X=np.mean(IMG,-1)
print('initial x=',X.shape)
X=np.reshape(X,(167,-1))
print('final x=',X.shape)


avgX = np.mean(X,axis=1)

b=np.tile(avgX,(X.shape[1],1)).T
X = X - np.tile(avgX,(X.shape[1],1)).T


U,S,VT=np.linalg.svd(X,full_matrices=False)

testIMG=X[157,:]
testIMG=np.reshape(testIMG,(883,900))
U1,S1,V1T=np.linalg.svd(testIMG,full_matrices=False)
S1=np.diag(S1)
j=0
plt.figure()
plt.rcParams['figure.figsize']=[16,8]
IMG=np.zeros((883,6*900))
for r in (10,20,50,100,120,150):
    img_approx = U[:,:r] @ S1[0:r,:r] @ VT[:r,:]
    #plt.figure(j+1)
    if j==0:
        IMG[:883,:900]=np.reshape(img_approx[157,:],(883,900))

    if j==1:
        IMG[:883,900:2*900]=np.reshape(img_approx[157,:],(883,900))
    if j==2:
        IMG[:883,2*900:3*900]=np.reshape(img_approx[157,:],(883,900))
    if j==3:
        IMG[:883,3*900:4*900]=np.reshape(img_approx[157,:],(883,900))
    if j==4:
        IMG[:883,4*900:5*900]=np.reshape(img_approx[157,:],(883,900))
    if j==5:
        IMG[:883,5*900:6*900]=np.reshape(img_approx[157,:],(883,900))
    j=j+1

img = plt.imshow(IMG)
img.set_cmap('viridis') #('gray')
plt.axis('on')
plt.title('1)10 2)20 3)50 4) 100 5) 120 6)150')
plt.show()

## ploting energies ##



X2=np.arange(len(S))
fig, ax= plt.subplots()
ax.plot(np.cumsum(S)/np.sum(S))
plt.title('energy distribution:')
plt.show()

## COMMENTS :
#2.)Columns of U represent the eigen flow basis in heirachial order
#3.) We find that as we increase 'r' value , the approximation to image gets better
#4.) Final plot shows the distribution of energy accross various principal components.
