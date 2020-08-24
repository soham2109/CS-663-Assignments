import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from math import ceil,floor
from numpy import zeros,zeros_like,array
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import c_


def coeffUpdate(input_image,x,y):
    p = input_image.copy()
    r,c = input_image.shape[:2]
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

def myBicubicInterpolation(input_image,row_ratio=3,col_ratio=2,cmap="gray"):
    rows,columns = input_image.shape[:2]
    new_cols = col_ratio*columns-1
    new_rows = row_ratio*rows-2
    output = zeros((new_rows,new_cols))
    #coeffs = coeffUpdate(input_image)
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
#             elif(r1>=0 and r1+2<rows and c1>=0 and c1+2<columns):
#                 a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33 = coeffUpdate(input_image,r1-1,c1-1)
#             elif(r1>=0 and r1+1<rows and c1>=0 and c1+1<columns):
#                 a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33 = coeffUpdate(input_image,r1-2,c1-2)
#             elif(r1>=0 and c1>=0):
#                 a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33 = coeffUpdate(input_image,r1-3,c1-3)
                output[row][col] = (a00 + a01 * y + a02 * y**2 + a03 * y**3) + (a10 + a11 * y + a12 * y**2 + a13 * y**3) * x + (a20 + a21 * y + a22 * y**2 + a23 * y**3) * x**2 + (a30 + a31 * y + a32 * y**2 + a33 * y**3) * x**3
    
    fig,axes = plt.subplots(1,2, constrained_layout=True, gridspec_kw={'width_ratios':[1,2]})
    axes[0].imshow(input_image,cmap)
    axes[0].axis("on")
    
    im = axes[1].imshow(output,cmap)
    axes[1].axis("on")
    
    cbar = fig.colorbar(im,ax=axes.ravel().tolist())
    #cbar.ax.set_yticklabels([0,255])
    plt.show()        

input_file = "../data/barbaraSmall.png"
input_image = mpimg.imread(input_file,format="png")*255
myBicubicInterpolation(input_image)
