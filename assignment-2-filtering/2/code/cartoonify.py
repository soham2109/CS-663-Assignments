import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros,zeros_like,exp,roll,array
from numpy.random import normal
import h5py
import cv2

np.random.seed(0)

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
    return result/wgt_sum

def plot_images(filename,ssp,sint,input_image,output_image,cmap="gray"):
    
    name = filename.split("/")[-1].split(".")[0]

    fig,axes = plt.subplots(1,2, constrained_layout=True)
    axes[0].imshow(input_image/np.max(input_image),cmap=cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image",fontsize=12)
    
    im = axes[1].imshow(output_image/np.max(output_image), cmap=cmap)
    axes[1].axis("on")
    axes[1].set_title("Bilateral Filtered Image",fontsize=12)

    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.35)
    plt.savefig("../images/"+name+str(ssp)+"_"+str(sint)+"Bilateral.png",bbox_inches="tight",pad=-1)
    plt.imsave("../images/"+name+str(ssp)+"_"+str(sint)+"BilateralOnly.png",output_image/np.max(output_image),cmap=cmap)
    plt.cla()


def myBilateralFiltering(filename,sigma_spatial,sigma_intensity):
    name = filename.split("/")[-1].split(".")[0]
    image = cv2.imread(filename)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    output = filter_bilateral(filename,array(image,dtype="float32"),sigma_spatial,sigma_intensity)
    for _ in range(3):
        output= filter_bilateral(filename,output,sigma_spatial,sigma_intensity)
        
    plot_images(filename,sigma_spatial,sigma_intensity,image,output)
    
    
if __name__=="__main__":
    filename = "../data/tom_cruise.jpg"
    myBilateralFiltering(filename,5,10)
    
