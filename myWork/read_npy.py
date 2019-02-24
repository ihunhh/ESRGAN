import numpy as np 
import tensorlayer as tl

#with np.load('vgg19.npy', encoding='latin1') as data:
npz = np.load('vgg19.npy', encoding='latin1').item()
w = np.asarray(npz['conv2_2'])
print(w)
npz.pop('fc6', None)
npz.pop('fc7', None)
npz.pop('fc8', None)
#print(np.dtype(w))
print(len(npz))
print(sorted(npz.keys()))
#print(npz.items())