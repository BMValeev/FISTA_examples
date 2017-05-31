#!/usr/bin/python3
import time
import sys
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
"""
Script can work only with grayscaled omage for a while. In future it will be modified and written as a class
"""
lambd=0.00001
alpha=10
Nit=200
def conv2(x, y, mode='same'):
	return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
	"""
	2D gaussian mask - should give the same result as MATLAB's
	fspecial('gaussian',[shape],[sigma])
	"""
	m,n = [(ss-1.)/2. for ss in shape]
	y,x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	return h

def soft(x,T):
	"""
	Function for soft thresholding the image vector
	"""
	eq=np.greater_equal(np.absolute(x), np.absolute(T))
	y=eq*(np.absolute(x)-np.absolute(T))
	return y

def fista_normal(y,H,Ht,lambd,alpha,Nit):
	"""
	Main iteration process of the optimization
	"""
	x=conv2(y, Ht)
	y_k=x
	t_1=1
	for i in range(0,Nit):
		x_old=x
		x=soft((y_k+(1/alpha)*conv2((y-conv2(y_k, H)), Ht)),lambd/alpha)
		t_0=t_1
		t_1=0.5+np.sqrt(0.25+(t_1*t_1))
		y_k=x+((t_0-1)/t_1)*(x-x_old)
	return x

kernel=matlab_style_gauss2D(shape=(7,7),sigma=2)
kernel_t=np.transpose(kernel)
im = Image.open("lena512.bmp")
p = np.array(im)
s=np.array(im)
p=conv2(p,kernel)
#p[:,:,0]=conv2(p[:,:,0],kernel)
#p[:,:,1]=conv2(p[:,:,1],kernel)
#p[:,:,2]=conv2(p[:,:,2],kernel)
r = p
tic = time.time()
r=255*fista_normal(p/255,kernel,kernel_t,lambd,alpha,Nit)
toc = time.time()
print( toc-tic, 'sec Elapsed')
#r[:,:,0]=255*fista_normal(p[:,:,0]/255,kernel,kernel_t,lambd,alpha,Nit)
#r[:,:,1]=255*fista_normal(p[:,:,1]/255,kernel,kernel_t,lambd,alpha,Nit)
#r[:,:,2]=255*fista_normal(p[:,:,2]/255,kernel,kernel_t,lambd,alpha,Nit)
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.imshow(s, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(132)
plt.imshow(p, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.imshow(r, cmap=plt.cm.gray)
plt.axis('off')
plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01,
                    left=0.01, right=0.99)
plt.show()
