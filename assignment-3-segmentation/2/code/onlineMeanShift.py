import cv2
import numpy as np
import sys,os

script_dir = sys.path[0] # This gives the directory in which our script is running on the system
img_path = os.path.join(script_dir, '../data/'+sys.argv[1])
img = cv2.imread(img_path) # Read image here


def im2double(im):
	#im1,im2,im3 = cv2.split(im)
	im1 = im[:,:,2]
	im2 = im[:,:,1]
	im3 = im[:,:,0]
	out1 = (im1.astype('float') - np.min(im1.ravel())) / (np.max(im1.ravel()) - np.min(im1.ravel()))
	out2 = (im2.astype('float') - np.min(im2.ravel())) / (np.max(im2.ravel()) - np.min(im2.ravel()))
	out3 = (im3.astype('float') - np.min(im3.ravel())) / (np.max(im3.ravel()) - np.min(im3.ravel()))
	out = cv2.merge((out1,out2,out3))
	return out


img = im2double(img)
gaussian_blur = cv2.GaussianBlur(img, (5,5), 1.0)
shp = img.shape
# row = shp[0]//4
# col = shp[1]//4
row,col = 128,128
newimg = cv2.resize(img,(row,col))



result1 = np.zeros((row,col))
result2 = np.zeros((row,col))
result3 = np.zeros((row,col))
h=0.1
sigma = 11.0 
count = 0
window_size = 7
padded = np.concatenate((np.concatenate((np.zeros((row,window_size,3)), newimg),axis=1), np.zeros((row,window_size,3))),axis=1)

padded = np.concatenate((np.concatenate((np.zeros((window_size,col+2*(window_size),3)),padded),axis=0), np.zeros((window_size, col+2*(window_size),3))),axis=0)
r,c,d = padded.shape
print(padded.shape)

#isotropic
#Spatial
spatial = np.zeros((2*window_size+1,2*window_size+1))
for is1 in range(2*window_size+1):
	for is2 in range(2*window_size+1):
		spatial[is1][is2] = ((is1-window_size)**2+(is2-window_size)**2)**0.5
spatial = np.exp(-(spatial/sigma)**2)


for idx1 in range(window_size,r-window_size):
	for idx2 in range(window_size,c-window_size):
		window = padded[idx1-window_size:idx1+window_size+1, idx2-window_size:idx2+window_size+1]
		#print(newimg[idx1][idx2])
		(x1,x2,x3) = newimg[idx1-window_size][idx2-window_size]
		N1 = 0.0 #numerator
		D1 = 1.0 #denominator
		N2 = 0.0 #numerator
		D2 = 1.0 #denominator
		N3 = 0.0 #numerator
		D3 = 1.0 #denominator
		
		
		for itern in range(20): # number of iterations
			for idx3 in range(2*window_size+1):
				for idx4 in range(2*window_size+1):
					(x_i1,x_i2,x_i3) = window[idx3][idx4]
					diff1 = abs(x1-x_i1)
					diff2 = abs(x2-x_i2)
					diff3 = abs(x3-x_i3)
					
					d1 = np.exp(-(diff1/h)**2)*spatial[idx3][idx4]
					n1 = x_i1*d1
					N1 += n1
					D1 += d1
					
					d2 = np.exp(-(diff2/h)**2)*spatial[idx3][idx4]
					n2 = x_i2*d2
					N2 += n2
					D2 += d2
					
					d3 = np.exp(-(diff3/h)**2)*spatial[idx3][idx4]
					n3 = x_i3*d3
					N3 += n3
					D3 += d3
					
			x1 = float(N1)/D1 # in each iteration, x changes
			x2 = float(N2)/D2 # in each iteration, x changes
			x3 = float(N3)/D3 # in each iteration, x changes
			
		result1[idx1-window_size][idx2-window_size] = x1
		result2[idx1-window_size][idx2-window_size] = x2
		result3[idx1-window_size][idx2-window_size] = x3
		
		if count%1000 == 0:
			print(count)
		count += 1
'''
cv2.imshow('Original Image',spatial)
print(spatial)
'''
result = cv2.merge((result1,result2,result3))
print(result)
cv2.imshow('Resultant Image',result)
cv2.imshow('Resized Image',newimg)

cv2.waitKey(0)
