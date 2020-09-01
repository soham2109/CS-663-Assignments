import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from numpy import zeros_like

from tqdm import tqdm
import cv2
from seaborn import distplot
from math import floor,ceil

def plot_hist(input_file,input_image,output_image,reference_image):
    
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    name = input_file.split(".")[2]
    plt.figure()
    plt.title("Normalized Histogram Plots for Images")
    ax = distplot(input_image,color='r',label ="Input Histogram",hist_kws={"alpha": 0.3, "linewidth": 1.5},bins=256,hist=False)
    ax = distplot(output_image,color="b",label ="Histogram Matched Histogram",hist_kws={"alpha": 0.3,"linewidth": 1.5},bins=256,hist=False)
    ax = distplot(reference_image,color='g',label ="Reference Histogram",hist_kws={"alpha": 0.3, "linewidth": 1.5},bins=256,hist=False)
    
    l1 = ax.lines[0]
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    ax.fill_between(x1,y1, color="red", alpha=0.3)
    l2 = ax.lines[1]
    x2 = l2.get_xydata()[:,0]
    y2 = l2.get_xydata()[:,1]
    ax.fill_between(x2,y2, color="blue", alpha=0.3)
    l3 = ax.lines[2]
    x3 = l3.get_xydata()[:,0]
    y3 = l3.get_xydata()[:,1]
    ax.fill_between(x3,y3, color="green", alpha=0.3)
    plt.legend()
    plt.savefig(".."+name+"HMHistogram.png",bbox_inches="tight",pad=-1)
    

def perform_masking(original,masking,r,c,d=3):
    """
    Masks the original image using the mask provided
    input : orig_image, mask, r(rows of orig_image), c(cols of orig_image), d(#channels of orig_image)
    output : masked image
    """
    orig = original.copy()
    mask = masking.copy()
    print(np.unique(mask))
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(orig)
    for i in range(3):
        for j in range(r):
            for k in range(c):
                orig[j,k,i] = (0 if mask[j,k,i]==0 else orig[j,k,i])
        
        
    plt.subplot(1,2,2)
    plt.imshow(orig)
    return orig

def calculate_CDF(array,maximum,r,c):
    """
    This function is used to calculate the CDF of the 2D image.
    array : For gray scale the whole image is the input and for RGB each of the color slices are the inputs.
    maximum : maximum pixel intensity of the 2D image
    output : the CDF of the 2D image
    """
    freqs = np.zeros((maximum+1,1))
    probf = np.zeros((maximum+1,1))
    cum = np.zeros((maximum+1,1))

    for i in range(r):
        for j in range(c):
            freqs[int(array[i][j])]+=1

    for i,j in enumerate(freqs):
        probf[i] = freqs[i]/(r*c)

    for i,j in enumerate(probf):
        for k in range(i):
            cum[i] += probf[k]
    return cum

def myHM(reference,reference_mask,target,target_mask):
    name = target.split(".")[2]
    
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    # READING THE REFERENCE IMAGE
    original_ref_image = cv2.imread(reference)
    original_ref_image = cv2.cvtColor(original_ref_image,cv2.COLOR_BGR2RGB)
    
    r1,c1,d1 = original_ref_image.shape
    # READING THE REFERENCE IMAGE MASK AND PRODUCING THE MASKED REFERENCE IMAGE
    original_ref_image_mask = cv2.imread(reference_mask)
    original_ref_image_mask = cv2.cvtColor(original_ref_image_mask,cv2.COLOR_BGR2RGB)
    original_ref_masked = perform_masking(original_ref_image,original_ref_image_mask,r1,c1,d1)
    
    # READING THE IMAGE TO BE HISTOGRAM MATCHED
    original_target_image = cv2.imread(target)
    original_target_image = cv2.cvtColor(original_target_image,cv2.COLOR_BGR2RGB)
    
    r2,c2,d2 = original_target_image.shape
    
    # READING THE MASK OF THE IMAGE TO BE MATCHED
    original_target_image_mask = cv2.imread(target_mask)
    original_target_image_mask = cv2.cvtColor(original_target_image_mask,cv2.COLOR_BGR2RGB)
    original_target_masked = perform_masking(original_target_image,original_target_image_mask,r2,c2,d2)
  
    # CREATING THE HSV TRANSFORM FOR THE MASKED IMAGES AND EXTRACTING THE V-CHANNEL
    ref_hsv = cv2.cvtColor(original_ref_masked,cv2.COLOR_RGB2HSV)
    ref_v = cv2.split(ref_hsv)[2]
    target_hsv = cv2.cvtColor(original_target_masked,cv2.COLOR_RGB2HSV)
    target_v = cv2.split(target_hsv)[2]
    
    ref_image = ref_hsv.copy()
    target_image = target_hsv.copy()
    
    # CUMULATIVE DISTRIBUTION OF THE IMAGES
    cum_ref = calculate_CDF(ref_v,int(np.max(ref_v)),r1,c1)
    cum_target = calculate_CDF(target_v,int(np.max(target_v)),r2,c2)
    
    transform_ref=[0 for i in range(256)]
    transform_tar=[0 for i in range(256)]

    for i in range(len(cum_target)):
        transform_tar[i] = floor(255*cum_target[i])
        transform_ref[i] = floor(255*cum_ref[i]) 

    transform ={}  
    
    for i in transform_tar:
        value=min(transform_ref, key=lambda x:abs(x-i))
        indx = transform_ref.index(value)
        original = transform_tar.index(i)
        transform[original]=indx
    
    for i in range(r1):
            for j in range(c1):
                 if(target_image[i,j,2] in transform):
                    target_image[i,j,2]=transform[target_image[i,j,2]]
    
    plot_hist(target,target_v,target_image[:,:,2],ref_v)
    
    # CONVERT BACK FROM HSV TO RGB
    target_image = cv2.cvtColor(target_image,cv2.COLOR_HSV2RGB)
    
    
    
    fig,axes = plt.subplots(1,3, constrained_layout=True)

    axes[0].imshow(original_ref_masked)
    axes[0].axis("on")
    axes[0].set_title("Reference Image")
    axes[1].imshow(original_target_masked)
    axes[1].axis("on")
    axes[1].set_title("Original Image")
    im = axes[2].imshow(target_image)
    axes[2].axis("on")
    axes[2].set_title("Histogram Matched")
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    plt.savefig(".."+name+"HistogramMatched.png",cmap="gray",bbox_inches="tight",pad=-1)
    
    plt.imsave(".." + name+"HM.png",target_image,cmap="gray")
    
