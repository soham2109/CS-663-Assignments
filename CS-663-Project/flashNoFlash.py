import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy.ndimage import convolve
from skimage.morphology import disk

def gauss_ker(k, sig):
	"""create a gaussian kernel with given size and standard deviation
	;param k; the kernel size
	:param sig: the standard deviation of the gaussian kernel
	"""
	x = np.linspace(-(k//2), (k//2), k)
	gx, gy = np.meshgrid(x, x)
	kernel = np.exp(-1*(gx**2 + gy**2)/(2*(sig**2)))
	return kernel


def _normalize(array):
	"""Min-Max Normalization of the array
	:param array: 2-D input image array
	:output : returns normalized arrays
	"""
	return (array - np.min(array))/(np.max(array)-np.min(array))

def normalize(array):
	"""Divide the array by 
	"""
	return array/np.max(array)

def bilateral(filename,input_image, sigma_spatial, sigma_intensity):
	"""
	Performs standard bilateral filtering of an input image. If padding is desired,
	img_in should be padded prior to calling

	inputs:- input_image       (ndarray) input image
		   - sigma_spatial      (float)   spatial gaussian standard deviation
		   - sigma_intensity      (float)   value gaussian standard. deviation
	outputs:-result      (ndarray) output bilateral-filtered image
	"""
	# make a simple Gaussian function taking the squared radius
	gaussian = lambda r2, sigma: np.exp(-0.5*r2/sigma**2 )
	#print(input_image.shape)
	input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)

	# define the window width to be the 2 time the spatial std. dev. to
	# be sure that most of the spatial kernel is actually captured
	win_width = int(3*sigma_spatial +1)
	wgt_sum = np.zeros_like(input_image).astype(np.float64)
	result  = np.zeros_like(input_image).astype(np.float64)
	out= np.zeros_like(input_image).astype(np.float64)
	
	for i in tqdm(range(input_image.shape[-1]),desc="Going through color channels"):
		norm_image = normalize(input_image[:,:,i])
		for shft_x in range(-win_width,win_width+1):
			for shft_y in range(-win_width,win_width+1):
				# compute the spatial contribution
				spatial = gaussian(shft_x**2+shft_y**2, sigma_spatial )
	
				# shift by the offsets to get image window
				window = np.roll(norm_image, [shft_y, shft_x], axis=[0,1])
	
				# compute the intensity contribution
				combined_filter = spatial*gaussian( (window-norm_image)**2, sigma_intensity )
	
				# result stores the mult. between combined filter and image window
				result[:,:,i] += window*combined_filter
				wgt_sum[:,:,i] += combined_filter
	out = normalize(result/wgt_sum)

	# normalize the result and return
	plt.imsave("outputImages/Bilateral_"+filename+"_"+str(sigma_spatial)+"_"+ str(sigma_intensity) + ".png" ,out,dpi=600)
	return out
	
	
	
def joint_bilateral(filename,flash_image,noflash_image,sigma_spatial,sigma_intensity):
	"""Joint Bilateral filter taking spatial gaussian from noFlash and intensity gaussian from flash image
	:param filename: a file name to save the joint bilateral
	"""
	# make a simple Gaussian function taking the squared radius
	gaussian = lambda r2, sigma: np.exp(-0.5*r2/sigma**2)
	flash_image = cv2.cvtColor(flash_image,cv2.COLOR_BGR2RGB)
	noflash_image = cv2.cvtColor(noflash_image,cv2.COLOR_BGR2RGB)

	# define the window width to be the 2 time the spatial std. dev. to
	# be sure that most of the spatial kernel is actually captured
	win_width = int(3*sigma_spatial +1)
	wgt_sum = np.zeros_like(flash_image).astype(np.float64)
	result  = np.zeros_like(flash_image).astype(np.float64)
	out= np.zeros_like(flash_image).astype(np.float64)
	
	
	for i in tqdm(range(flash_image.shape[-1]),desc="Going through color channels"):
		norm_flash_image = normalize(flash_image[:,:,i])
		norm_noflash_image = normalize(noflash_image[:,:,i])
		for shft_x in range(-win_width,win_width+1):
			for shft_y in range(-win_width,win_width+1):
				# compute the spatial contribution
				spatial = gaussian(shft_x**2+shft_y**2, sigma_spatial )
	
				# shift by the offsets to get image window
				window = np.roll(norm_flash_image, [shft_y, shft_x], axis=[0,1])
				window1 = np.roll(norm_noflash_image, [shft_y, shft_x], axis=[0,1])
				# compute the intensity contribution
				combined_filter = spatial*gaussian((window-norm_flash_image)**2, sigma_intensity )
	
				# result stores the mult. between combined filter and image window
				result[:,:,i] += window1*combined_filter
				wgt_sum[:,:,i] += combined_filter
	out = normalize(result/wgt_sum)
	# normalize the result and return
	plt.imsave("outputImages/JointBilateral_"+filename+"_"+str(sigma_spatial)+"_"+ str(sigma_intensity) + ".png" ,out,dpi=600)
	return out

	
def detail_transfer(name,flash_image,flash_bilateral,eps=0.02):
	# flash_bilateral = bilateral(name,flash_image,sigma_spatial,sigma_intensity)
	flash_image = cv2.cvtColor(flash_image.astype(np.float32),cv2.COLOR_BGR2RGB)
	detail = (normalize(flash_image) + eps)/(flash_bilateral + eps)
	#avg_detail = (detail[:,:,0] + detail[:,:,1]+detail[:,:,2])/3
	plt.imsave("outputImages/DetailLayer_"+name+".png", _normalize(detail),dpi=600)
	#plt.imsave("outputImages/DetailLayer_"+name+"_" + str(sigma_spatial)+"_"+str(sigma_intensity)+".png",avg_detail,cmap="gray")
	return detail

def calculate_mask(name,flash_image,noflash_image):

	#luminance
	Y_flash = cv2.cvtColor(flash_image,cv2.COLOR_BGR2GRAY)
	Y_noflash = cv2.cvtColor(flash_image,cv2.COLOR_BGR2GRAY)

	#shadow
	diff_image = Y_flash - Y_noflash
	shadow_mask = np.zeros(diff_image.shape, np.uint8)
	
	thr1 = -0.05
	thr2 = -0.2 
	shadow_mask[(diff_image > thr2) & (diff_image < thr1)] = 1
	shadow_mask[(diff_image > 0.65) & (diff_image < 0.7)] = 1

	#Specularity
	max_flash = 0.95 * (np.max(Y_flash) - np.min(Y_flash))
	# shadow_mask=np.zeros_like(diff_image)
	shadow_mask[Y_flash>max_flash] = 1 
	
	# dilation and erosion using circle elements
	se1 = disk(2)
	se2 = disk(6)
	se3 = disk(4)
	maskff = np.zeros((shadow_mask.shape[0]+2,shadow_mask.shape[1]+2), np.uint8)
	shadow_mask = cv2.erode(shadow_mask, se1, iterations = 1)
	cv2.floodFill(shadow_mask, maskff, (0,0), 1)
	maskff = cv2.dilate(maskff, se2,  iterations = 1)
	maskff = cv2.erode(maskff, se3, iterations = 1)
	maskff = maskff.astype('double')
	maskff = cv2.filter2D(maskff, -1, gauss_ker(3,3))

	plt.imsave("outputImages/CombinedMask_"+name+".png",maskff,cmap="gray",dpi=600)	
	return maskff


def flash_adjust(flash_image, noflash_image, alpha):
	
	flash_image = cv2.cvtColor(flash_image,cv2.COLOR_BGR2YCR_CB)
	noflash_image = cv2.cvtColor(noflash_image,cv2.COLOR_BGR2YCR_CB)
	
	adjust_img = np.zeros(noflash_image.shape).astype('double')
	adjust_img = alpha*flash_image + (1-alpha)*noflash_image
	
	adjust_img = adjust_img.astype(np.uint8)
	adjust_img = cv2.cvtColor(adjust_img, cv2.COLOR_YCR_CB2RGB)
	
	return adjust_img

def remove_noise(name, noflash_bilateral, f_detail, joint_b, mask):
    mask = mask[:-2,:-2]
    final = np.dstack(((1-mask), (1-mask), (1-mask)))*(joint_b*f_detail) + np.dstack((mask, mask, mask))*(noflash_bilateral)
    plt.figure()
    plt.imshow("outputImages/NoiseRemoved_"+name+".png",final)
    plt.axis("off")
    plt.savefig("outputImages/NoiseRemoved_"+name+".png", bbox_inches='tight', dpi=600)
    # cv2.imwrite("outputImages/NoiseRemoved_"+name+".png",final)
    return final


def white_balancing(flash_image, noflash_image, ):
	if opt==1:
		scaling=(255/246,255/169,255/87)
	else:
		pass
	pass

if __name__=="__main__":
	input_filename_flash = "flash_data_JBF_Detail_transfer/potsdetail_00_flash.tif"
	input_filename_noflash = "flash_data_JBF_Detail_transfer/potsdetail_01_noflash.tif"
	
	# input_filename_flash = "flash_data_JBF_Detail_transfer/lamp_00_flash.tif"
	# input_filename_noflash = "flash_data_JBF_Detail_transfer/lamp_01_noflash.tif"

	name = input_filename_noflash.split("/")[-1].split("_")[0]

	noflash_image = cv2.imread(input_filename_noflash)
	flash_image = cv2.imread(input_filename_flash)
	#image_noflash = cv2.resize(image_noflash,(100,100),interpolation = cv2.INTER_AREA)
	#image_flash = cv2.resize(image_flash,(100,100),interpolation = cv2.INTER_AREA)
	
		
	bilateral_flash = bilateral(name+"flash",flash_image,3,0.06)
	bilateral_noflash = bilateral(name+'no_flash',noflash_image,3,0.06)
	joint_bilateral_out = joint_bilateral(name, flash_image, noflash_image,3, 0.06)

	mask = calculate_mask(name, flash_image, noflash_image)
	f_detail = detail_transfer(name,flash_image,bilateral_flash,0.02)

	final = remove_noise(name, bilateral_noflash, f_detail, joint_bilateral_out, mask)
	without_mask = f_detail*joint_bilateral_out
	plt.imsave("outputImages/Without_Mask_"+name+".png",_normalize(without_mask))
