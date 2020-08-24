import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

from math import ceil,floor
from numpy import zeros,zeros_like,array
from mpl_toolkits.axes_grid1 import ImageGrid

def myNearestNeighbourInterpolation(input_image,row_ratio=3,col_ratio=2,cmap="gray"):
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
        r1 = (floor(r) if (r%1)<0.5 else ceil(r))
        for col in range(new_cols):
            c = col/col_ratio
            c1 = (floor(c) if (c%1)<0.5 else ceil(c))
            output[row][col] = input_image[r1][c1]

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
myNearestNeighbourInterpolation(input_image)
