# IMPORTING MODULES FOR BASIC COMPUTATION
import numpy as np 
import matplotlib.pyplot as plt # for plotting
import matplotlib.image as mpimg # for image reading
import matplotlib as mpl

from numpy import zeros,zeros_like,array

def myShrinkImageByFactorD(input_file,d,cmap="gray"):
    """
    d : List of shrinkage factors
    Shrink the image size by a factor of d along each dimension using image subsampling by
    sampling / selecting every d -th pixel along the rows and columns.
    usage : myShrinkImageByFactorD([2,3])
    input : <input_image_path>,D (list of shrink factors D)
    output : None
    Saves the shrinked image data in the ../data folder
    """
    
    name = input_file.split("."[2]
    input_image = mpimg.imread(input_file,format="png")
    num_plots = len(d)+1
    
    width = input_image.shape[0]
    height = input_image.shape[1]
    output_images = []
    
    fig,axes = plt.subplots(1,num_plots, constrained_layout=True)
    
    # PLOTTING PARAMETERS
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)

    axes[0].imshow(input_image,cmap=cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    
    count = 0

    for i in d:
        count = count + 1
        new_width = int(width/i)
        new_height = int(height/i)
        output = zeros((new_width,new_height))
        for W in range(new_width):
            for H in range(new_height):
                output[W][H] = input_image[W*i][H*i]
        output_images.append(output)

        im = axes[count].imshow(output, cmap=cmap)
        axes[count].axis("on")
        axes[count].set_title("Shrink by Factor"+str(i))

    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    plt.savefig(".."+name+"Shrink.png",bbox_inches="tight",pad=-1,cmap=cmap)
    
    for i in range(len(d)):
        plt.imsave(".."+name+"ShrinkByFactor"+str(d[i])+".png",output_images[i],cmap=cmap)
