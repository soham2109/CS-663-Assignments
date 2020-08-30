import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numpy import zeros,zeros_like,array,c_
from math import floor,ceil


def coeffUpdate(input_image,x,y):
    """
    This function calculates the 16 coefficients necessary for calculating the 
    bicubic interpolation equations.
    input : input_image , present co-ordinates (x=row,y=col)
    output : the 16 coefficients a00 to a33
    """
    p = input_image.copy()
    r,c = input_image.shape
    q = zeros((r,4))
    s = zeros((4,c+4))
    p = np.concatenate((p,q), axis=1)
    p = np.concatenate((p,s), axis=0)
    
    for i in range(4):
        p[r+i][:] = input_image[r-1][c-1]
        p[:][c+i] = input_image[r-1][c-1]
    
    a00 = p[x+1][y+1]
    a01 = -.5*p[x+1][y+0] + .5*p[x+1][y+2]
    a02 = p[x+1][y+0] - 2.5*p[x+1][y+1] + 2*p[x+1][y+2] - .5*p[x+1][y+3]
    a03 = -.5*p[x+1][y+0] + 1.5*p[x+1][y+1] - 1.5*p[x+1][y+2] + .5*p[x+1][y+3]
    a10 = -.5*p[x+0][y+1] + .5*p[x+2][y+1]
    a11 = .25*p[x+0][y+0] - .25*p[x+0][y+2] - .25*p[x+2][y+0] + .25*p[x+2][y+2]
    a12 = -.5*p[x+0][y+0] + 1.25*p[x+0][y+1] - p[x+0][y+2] + .25*p[x+0][y+3] + .5*p[x+2][y+0] - 1.25*p[x+2][y+1] + p[x+2][y+2] - .25*p[x+2][y+3]
    a13 = .25*p[x+0][y+0] - .75*p[x+0][y+1] + .75*p[x+0][y+2] - .25*p[x+0][y+3] - .25*p[x+2][y+0] + .75*p[x+2][y+1] - .75*p[x+2][y+2] + .25*p[x+2][y+3]
    a20 = p[x+0][y+1] - 2.5*p[x+1][y+1] + 2*p[x+2][y+1] - .5*p[x+3][y+1]
    a21 = -.5*p[x+0][y+0] + .5*p[x+0][y+2] + 1.25*p[x+1][y+0] - 1.25*p[x+1][y+2] - p[x+2][y+0] + p[x+2][y+2] + .25*p[x+3][y+0] - .25*p[x+3][y+2]
    a22 = p[x+0][y+0] - 2.5*p[x+0][y+1] + 2*p[x+0][y+2] - .5*p[x+0][y+3] - 2.5*p[x+1][y+0] + 6.25*p[x+1][y+1] - 5*p[x+1][y+2] + 1.25*p[x+1][y+3] + 2*p[x+2][y+0] - 5*p[x+2][y+1] + 4*p[x+2][y+2] - p[x+2][y+3] - .5*p[x+3][y+0] + 1.25*p[x+3][y+1] - p[x+3][y+2] + .25*p[x+3][y+3]
    a23 = -.5*p[x+0][y+0] + 1.5*p[x+0][y+1] - 1.5*p[x+0][y+2] + .5*p[x+0][y+3] + 1.25*p[x+1][y+0] - 3.75*p[x+1][y+1] + 3.75*p[x+1][y+2] - 1.25*p[x+1][y+3] - p[x+2][y+0] + 3*p[x+2][y+1] - 3*p[x+2][y+2] + p[x+2][y+3] + .25*p[x+3][y+0] - .75*p[x+3][y+1] + .75*p[x+3][y+2] - .25*p[x+3][y+3]
    a30 = -.5*p[x+0][y+1] + 1.5*p[x+1][y+1] - 1.5*p[x+2][y+1] + .5*p[x+3][y+1]
    a31 = .25*p[x+0][y+0] - .25*p[x+0][y+2] - .75*p[x+1][0] + .75*p[x+1][2] + .75*p[x+2][0] - .75*p[x+2][2] - .25*p[x+3][0] + .25*p[x+3][2]
    a32 = -.5*p[x+0][y+0] + 1.25*p[x+0][y+1] - p[x+0][y+2] + .25*p[x+0][y+3] + 1.5*p[x+1][y+0] - 3.75*p[x+1][y+1] + 3*p[x+1][y+2] - .75*p[x+1][y+3] - 1.5*p[x+2][y+0] + 3.75*p[x+2][y+1] - 3*p[x+2][y+2] + .75*p[x+2][y+3] + .5*p[x+3][y+0] - 1.25*p[x+3][y+1] + p[x+3][y+2] - .25*p[x+3][y+3]
    a33 = .25*p[x+0][y+0] - .75*p[x+0][y+1] + .75*p[x+0][y+2] - .25*p[x+0][y+3] - .75*p[x+1][y+0] + 2.25*p[x+1][y+1] - 2.25*p[x+1][y+2] + .75*p[x+1][y+3] + .75*p[x+2][y+0] - 2.25*p[x+2][y+1] + 2.25*p[x+2][y+2] - .75*p[x+2][y+3] - .25*p[x+3][y+0] + .75*p[x+3][y+1] - .75*p[x+3][y+2] + .25*p[x+3][y+3]
    
    return ([a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33])

def myBicubicInterpolation(input_file,row_ratio=3,col_ratio=2,cmap="gray",region=[]):
    """
    inputs : <input_image_path>,row-ratio, col-ratio,cmap(optional),region(optional)
    output = None
    Saves the bi-cubic Interpolated image to the ../data folder
    """
    input_image = mpimg.imread(input_file,format="png")
    name = input_file.split(".")[2]
    if len(region)!=0:
        input_image = input_image[region[0]:region[1],region[2]:region[3]]
        name="data/region"
    
    rows,columns = input_image.shape
    new_cols = col_ratio*columns-1
    new_rows = row_ratio*rows-2
    output = zeros((new_rows,new_cols))

    # PLOTTING PARAMETERS
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)

    for row in range(new_rows):
        r = (row/row_ratio)
        x = r%1
        r1 = floor(r)
        for col in range(new_cols):
            c = col/col_ratio
            y = c%1
            c1 = floor(c)
            if (r1>=0 and c1>=0):
                a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33 = coeffUpdate(input_image,r1,c1)
                output[row][col] = (a00 + a01 * y + a02 * y**2 + a03 * y**3) + (a10 + a11 * y + a12 *y**2 + a13 * y**3) * x + (a20 + a21 * y + a22 * y**2 + a23 * y**3)*x**2 + (a30 + a31 * y + a32 * y**2 + a33 * y**3) * x**3
    
    fig,axes = plt.subplots(1,2, constrained_layout=True, gridspec_kw={'width_ratios':[1,2]})
    axes[0].imshow(input_image,cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    im = axes[1].imshow(output,cmap)
    axes[1].axis("on")
    axes[1].set_title("Bicubic Interpolated")
    
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    plt.savefig(".."+name+"BicubicInterpolated.png",cmap=cmap,bbox_inches="tight",pad=-1)
    
    plt.imsave(".."+name+"Bicubic.png",output,cmap=cmap)       
