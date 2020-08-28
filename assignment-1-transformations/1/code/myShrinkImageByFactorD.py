import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

from numpy import zeros,zeros_like,array
from mpl_toolkits.axes_grid1 import ImageGrid


def myShrinkImageByFactorD(d,cmap="gray"):
    """
    d : List of 
    Shrink the image size by a factor of d along each dimension using image subsampling by
    sampling / selecting every d -th pixel along the rows and columns.
    This is the function that takes in the mutliples D to resize the image
    along with a colormap
    """
    input_file = "../data/circles_concentric.png"
    input_image = mpimg.imread(input_file,format="png")
    num_plots = len(d)+1
    
    width = input_image.shape[0]
    height = input_image.shape[1]
    
    fig,axes = plt.subplots(1,num_plots, constrained_layout=True, gridspec_kw={'width_ratios':[4,2,1]})
    
    #vmin = 0
    #vmax = 255

    axes[0].imshow(input_image,cmap=cmap)
    axes[0].axis("on")
    
    count = 0

    for i in d:
        count = count + 1
        new_width = int(width/i)
        new_height = int(height/i)
        output = zeros((new_width,new_height))
        for W in range(new_width):
            for H in range(new_height):
                #if W%i==0 and H%i==0:
                    #new_width = int(W*i)
                    #new_height = int(H*i)
                output[W][H] = input_image[W*i][H*i]

        im = axes[count].imshow(output, cmap=cmap)
        axes[count].axis("on")

    #plt.subplots_adjust(right=0.8)
    #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),ticks=[0,1],shrink=0.45)
    cbar.ax.set_yticklabels([0,255])
    plt.show()

myShrinkImageByFactorD([2,3])
