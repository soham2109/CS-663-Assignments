import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros,zeros_like,exp,roll,array
from numpy.random import normal
import h5py
import cv2

np.random.seed(0)

def read_file(filename):
    f = h5py.File(filename,"r")
    out = f.get('imageOrig')
    out = array(out)

    return (out*255.0/np.max(out))

def truncate(array):
    """
    if any pixel has value > 255 this makes it 255
    and if any pixel is <0 this makes it 0
    """
    r,c = array.shape
    for i in range(r):
        for j in range(c):
            if array[i,j]>255:
                array[i,j] = 255
            elif array[i,j]<0:
                array[i,j] = 0
    return array

def add_noise(image):
    out = image
    noise = normal(size=image.shape,scale=0.05*np.max(image))
    return truncate(out+noise)

def filter_bilateral(filename,input_image, sigma_spatial, sigma_intensity):
    """
    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    inputs:- input_image       (ndarray) input image
           - sigma_spatial      (float)   spatial gaussian standard deviation
           - sigma_intensity      (float)   value gaussian standard. deviation
    outputs:-result      (ndarray) output bilateral-filtered image
    """

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: exp(-0.5*r2/sigma**2 )

    # define the window width to be the 2 time the spatial std. dev. to
    # be sure that most of the spatial kernel is actually captured
    win_width = int(3*sigma_spatial +1)

    wgt_sum = zeros_like(input_image)
    result  = zeros_like(input_image)

    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial contribution
            spatial = gaussian(shft_x**2+shft_y**2, sigma_spatial )

            # shift by the offsets to get image window
            window = roll(input_image, [shft_y, shft_x], axis=[0,1])

            # compute the intensity contribution
            combined_filter = spatial*gaussian( (window-input_image)**2, sigma_intensity )

            # result stores the mult. between combined filter and image window
            result += window*combined_filter
            wgt_sum += combined_filter

    # normalize the result and return
    plt.imsave("../images/GaussianMask_"+filename+"_"+str(sigma_spatial)+"_"+ str(sigma_intensity) + ".png" ,wgt_sum,cmap="gray")
    return result/wgt_sum

def plot_images(filename,ssp,sint,input_image,noisy_image,output_image,cmap="gray"):
    
    name = filename.split("/")[-1].split(".")[0]

    fig,axes = plt.subplots(1,3, constrained_layout=True)
    axes[0].imshow(input_image/np.max(input_image),cmap=cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image",fontsize=12)
    
    axes[1].imshow(noisy_image/np.max(noisy_image),cmap=cmap)
    axes[1].axis("on")
    axes[1].set_title("Noisy Image",fontsize=12)

    im = axes[2].imshow(output_image/np.max(output_image), cmap=cmap)
    axes[2].axis("on")
    axes[2].set_title("Bilateral Filtered Image",fontsize=12)

    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.35)
    plt.savefig("../images/"+name+str(ssp)+"_"+str(sint)+"Bilateral.png",bbox_inches="tight",pad=-1)
    plt.imsave("../images/"+name+str(ssp)+"_"+str(sint)+"BilateralOnly.png",output_image/np.max(output_image),cmap=cmap)
    plt.cla()

def RMSD(A,B):
    r,c = A.shape
    A = A/np.max(A)
    B = B/np.max(B)
    total = np.sum(np.square(A-B))
    return np.sqrt(total/(r*c))

def myBilateralFiltering(filename,sigma_spatial,sigma_intensity):
    name = filename.split("/")[-1].split(".")[0]
    if filename.endswith(".mat"):
        image = read_file(filename)
    else:
        image = cv2.imread(filename,0)
    #print(image)
    noisy_image = add_noise(image)
    
    bilateral = filter_bilateral(name,noisy_image,sigma_spatial,sigma_intensity)
    plot_images(filename,sigma_spatial,sigma_intensity,image,noisy_image,bilateral)
    print("RMSD of {} image for Sigma_spatial: {} and Sigma_intensity: {} is {}".format(name,sigma_spatial,sigma_intensity,RMSD(image,bilateral)))
