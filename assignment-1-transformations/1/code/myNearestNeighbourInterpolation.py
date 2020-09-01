import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numpy import zeros,zeros_like,array
from math import floor,ceil


def myNearestNeighbourInterpolation(input_file,cmap="gray",region=[]):
    """
    input : <input_file_path>,cmap(optional),region (optional)
    output : None
    Saves the Nearest Neighbour interpolated images to the ../data folder.
    """
    name = input_file.split(".")[2].split("/")[2]
    input_image = mpimg.imread(input_file,format="png")
    
    if len(region)!=0:
        input_image = input_image[region[0]:region[1],region[2]:region[3]]
        name="region"
    
    # PLOTTING PARAMETERS
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    rows,columns = input_image.shape
    
    new_cols = 2*columns-1
    new_rows = 3*rows-2
    
    row_ratio = ceil(new_rows/rows)
    col_ratio = ceil(new_cols/columns)
    
    output = zeros((new_rows,new_cols))
    
    for row in range(new_rows):
        r = row/row_ratio
        r1 = (floor(r) if (r%1)<0.5 else ceil(r))
        for col in range(new_cols):
            c = col/col_ratio
            c1 = (floor(c) if (c%1)<0.5 else ceil(c))
            output[row][col] = input_image[r1][c1]
            
    fig,axes = plt.subplots(1,2, constrained_layout=True, gridspec_kw={'width_ratios':[1,2]})
    axes[0].imshow(input_image,cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    im = axes[1].imshow(output,cmap)
    axes[1].axis("on")
    axes[1].set_title("Nearest Neighbor Interpolated")
    
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    plt.savefig("../images/"+name+"NearestNeighbor.png",cmap=cmap,bbox_inches="tight",pad=-1)
    
    plt.imsave("../images/"+name+"NN.png",output,cmap=cmap)
