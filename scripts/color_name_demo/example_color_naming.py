import numpy as np
from keras.preprocessing import image
import h5py
import scipy.io
import matplotlib.pyplot as plt

# f = h5py.File('trackers/w2crs.mat')
# w2c = f['w2crs'].value

w2c = scipy.io.loadmat('w2c.mat')
w2c = w2c['w2c']

img_rgb = image.load_img('car.jpg')
img_rgb = image.img_to_array(img_rgb)

RR = img_rgb[:, :, 0]
GG = img_rgb[:, :, 1]
BB = img_rgb[:, :, 2]

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
# 'F' ,eams to flatten in column-major
index_im = np.floor(np.ndarray.flatten(RR, 'F')/8.) + \
           32 * np.floor(np.ndarray.flatten(GG, 'F')/8.) + \
           32 * 32 * np.floor(np.ndarray.flatten(BB, 'F')/8.)

# index_im = np.floor(np.ndarray.flatten(RR)/8.) + \
#            32 * np.floor(np.ndarray.flatten(GG)/8.) + \
#            32 * 32 * np.floor(np.ndarray.flatten(BB)/8.)


color_values = np.array([[0, 0, 0], [0, 0, 1], [.5, .4, .25], [.5, .5, .5], [0, 1, 0],
                         [1, .8, 0], [1, .5, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]])

w2cM = np.argmax(w2c, axis=1)

out2 = np.reshape(w2cM[index_im.astype('int')], (img_rgb.shape[0], img_rgb.shape[1]), 'F')\

out = img_rgb
for jj in range(img_rgb.shape[0]):
    for ii in range(img_rgb.shape[1]):
        out[jj,ii] = color_values[out2[jj,ii]]*255

plt.imshow(out/255.)
plt.waitforbuttonpress()