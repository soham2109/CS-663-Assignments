import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

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
    input : the input image, cmap(optional)
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
        
        for i in range(r):
            for j in range(c):
                new_image[i,j] = truncateHE(cum[int(new_input[i][j])]*maximum)
         
    ## For RGB, first convert to LAB and operate on L-channel and convert back to RGB
    else:
        input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(input_image,cv2.COLOR_RGB2LAB)
        output_image = zeros_like(image_lab)
        l,a,b = cv2.split(image_lab)
        l_copy = l.copy()
        maximum = int(np.max(l))
        cum = calculate_CDF(l,maximum,r,c)
        for i in range(r):
            for j in range(c):
                l_copy[i,j] = truncateHE(cum[int(l[i,j])]*maximum)
                
        output_image[:,:,0] = l_copy
        output_image[:,:,1] = a
        output_image[:,:,2] = b
        
        output_image = cv2.cvtColor(output_image,cv2.COLOR_LAB2RGB)
        
        
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
        
        
input_files = ["../data/chestXray.png","../data/barbara.png",
               "../data/statueForegroundMasked.png","../data/church.png",
               "../data/canyon.png","../data/TEM.png"]
for i in input_files:
    myHE(i)