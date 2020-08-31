import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from math import floor
import cv2
from seaborn import distplot

def plot_hist(input_file,input_image,output_image):
    """
    input : input_file_path, input_image, output_image
    output : saves the histograms for both the images for comparison
    dependencies : seaborn, numpy, matplotlib
    """
    name = input_file.split(".")[2]
    plt.figure()
    plt.title("Normalized Histogram Plots for Images")
    ax = distplot(input_image,color='r',label ="Input Histogram",hist_kws={"alpha": 0.3, "linewidth": 1.5},bins=256,hist=False)
    ax = distplot(output_image,color="b",label ="Histogram Equalized Histogram",hist_kws={"alpha": 0.3,"linewidth": 1.5},bins=256,hist=False)
    l1 = ax.lines[0]
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    ax.fill_between(x1,y1, color="red", alpha=0.3)
    l2 = ax.lines[1]
    x2 = l2.get_xydata()[:,0]
    y2 = l2.get_xydata()[:,1]
    ax.fill_between(x2,y2, color="blue", alpha=0.3)
    plt.legend()
    plt.savefig(".."+name+"HEHistogram.png",bbox_inches="tight",pad=-1)
    
    
def truncateHE(array):
    """
    This function truncates the array values to check whether the values
    are within range of [0,255]
    """
    if array<0:
        array = 0
    elif array>255.0:
        array = 255.0
    return array

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

def myHE(input_file,cmap="gray"):
    """
    This is the Histogram Equalization Function.
    input : the input image
    output : None
    Saves Histogram Equalized image
    """
    ## SETTING FONT-SIZE FOR PLOTTING
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    name = input_file.split(".")[2]
    input_image = cv2.imread(input_file)
    
    new_image = np.zeros_like(input_image)
    
    d = 1
    if len(input_image.shape)>2:
        r,c,d = input_image.shape
    else:
        r,c = input_image.shape
    
    if d==1:
        new_input = input_image
        maximum = int(np.max(new_input))
        cum = calculate_CDF(new_input,maximum,r,c)
        
        for i in tqdm(range(r)):
            for j in range(c):
                new_image[i,j] = truncateHE(cum[int(new_input[i][j])]*maximum)
        plot_hist(input_file,input_image,new_image)
                
    else:
        input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(input_image,cv2.COLOR_RGB2HSV)
        output_image = zeros_like(hsv_image)
        h,s,v = cv2.split(hsv_image)
        v_copy = v.copy()
        maximum = int(np.max(v))
        cum = calculate_CDF(v,maximum,r,c)
        for i in tqdm(range(r)):
            for j in range(c):
                v_copy[i,j] = truncateHE(cum[int(v[i,j])]*maximum)
                
        plot_hist(input_file,input_image,v_copy)
                
        output_image[:,:,2] = v_copy
        output_image[:,:,0] = h
        output_image[:,:,1] = s
        
        output_image = cv2.cvtColor(output_image,cv2.COLOR_HSV2RGB)
        
        
    fig,axes = plt.subplots(1,2, constrained_layout=True)
    

    axes[0].imshow(input_image,cmap=cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image")  
    
    im = axes[1].imshow(output_image, cmap=cmap)
    axes[1].axis("on")
    axes[1].set_title("Histogram Equalized")
    
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.35)
    plt.savefig(".."+name+"HistEq.png",bbox_inches="tight",pad=-1)
    
    if d==3:
        plt.imsave(".." + name+"HE.png",output_image)
    else:
        plt.imsave(".." + name+"HE.png",output_image,cmap=cmap)