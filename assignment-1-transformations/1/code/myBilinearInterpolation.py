import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numpy import zeros,zeros_like,array

from math import floor,ceil

def myBilinearInterpolation(input_file,cmap="gray",region=[]):
    """
    input = <input_file_path>,cmap(optional),region(optional)
    output = None
    Saves the bilinear Interpolated image to the ../images folder
    """
    input_image = mpimg.imread(input_file,format="png")
    name = input_file.split(".")[2].split("/")[2]
    
    # PLOTTING PARAMETERS
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    if len(region)!=0:
        input_image = input_image[region[0]:region[1],region[2]:region[3]]
        name="region"
    
    rows,columns = input_image.shape
    new_cols = 2*columns-1
    new_rows = 3*rows-2
    row_ratio = ceil(new_rows/rows)
    col_ratio = ceil(new_cols/columns)
    
    output = np.zeros((new_rows,new_cols))
    
    for row in range(new_rows):
        r = row/row_ratio
        r1 = floor(r)
        r2 = ceil(r)
        for col in range(new_cols):
            c = col/col_ratio
            c1 = floor(c)
            c2 = ceil(c)
            if(r1<=rows and r2<=rows and c1<=columns and c2<=columns):
                bottom_left = input_image[r1][c1]
                bottom_right = input_image[r2][c1]
                top_left = input_image[r1][c2]
                top_right = input_image[r2][c2]
                output[row][col] = bottom_right*(r%1)*(1-(c%1)) + bottom_left*(1-r%1)*(1-c%1) + top_right*(r%1)*(c%1) + top_left*(1-r%1)*(c%1)
    
    fig,axes = plt.subplots(1,2, constrained_layout=True, gridspec_kw={'width_ratios':[1,2]})
    axes[0].imshow(input_image,cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    im = axes[1].imshow(output,cmap)
    
    axes[1].axis("on")
    axes[1].set_title("Bilinear Interpolated Image")
    
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    
    # SAVING THE IMAGE WITH INTERPOLATED AND ORIGINAL
    plt.savefig("../images/"+name+"BilinearInterpolation.png",cmap=cmap,bbox_inches="tight",pad=-1)
    
    # SAVING THE INTERPOLATED IMAGE
    plt.imsave("../images/"+name+"Bilinear.png",output,cmap=cmap)
            
