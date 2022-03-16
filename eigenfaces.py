import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 18})

mat_contents = scipy.io.loadmat(os.path.join( 'allFaces.mat'))
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])
print('faces=',faces,faces.shape)
print('nfaces=',nfaces,nfaces.shape)
print(np.sum(nfaces))

allPersons = np.zeros((n * 6, m * 6))
count = 0

for j in range(6):
    for k in range(6):
        allPersons[j * n: (j + 1) * n, k * m: (k + 1) * m] = np.reshape(faces[:, np.sum(nfaces[:count])], (m, n)).T
        count += 1

img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
#plt.show()

