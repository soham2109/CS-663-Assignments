import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from math import ceil,floor
from numpy import zeros,zeros_like,array
from mpl_toolkits.axes_grid1 import ImageGrid

def myBilinearInterpolation(input_image,row_ratio=3,col_ratio=2,cmap="gray"):
#     input_file = "../data/barbaraSmall.png"
#     input_image = mpimg.imread(input_file,format="png")
    rows = input_image.shape[0]
    columns = input_image.shape[1]
    new_cols = col_ratio*columns-1
    new_rows = row_ratio*rows-2
    #row_ratio = ceil(new_rows/rows)
    #col_ratio = ceil(new_cols/columns)
    output = zeros((new_rows,new_cols))
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
                bottom_right = input_image[r1][c2]
                top_left = input_image[r2][c1]
                top_right = input_image[r2][c2]
                output[row][col] = bottom_right*(r%1)*(1-(c%1)) + bottom_left*(1-r%1)*(1-c%1)+top_right*(r%1)*(c%1) + top_left*(1-r%1)*(c%1)

    fig,axes = plt.subplots(1,2, constrained_layout=True, gridspec_kw={'width_ratios':[1,2]})
    axes[0].imshow(input_image,cmap)
    axes[0].axis("on")

    im = axes[1].imshow(output,cmap)
    axes[1].axis("on")

    cbar = fig.colorbar(im,ax=axes.ravel().tolist())
    #cbar.ax.set_yticklabels([0,255])
    plt.show()

input_file = "../data/barbaraSmall.png"
input_image = mpimg.imread(input_file,format="png")*255.0
myBilinearInterpolation(input_image)
