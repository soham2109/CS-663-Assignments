import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

from numpy import zeros,zeros_like,array
from mpl_toolkits.axes_grid1 import ImageGrid
from math import cos,sin,pi,radians

def myImageRotation(input_image,theta,cmap="gray"):
    theta = radians(theta)
    translation_matrix = array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    rows,columns = input_image.shape[:2]
    x_mid,y_mid = (rows-1)/2,(columns-1)/2
    new_image = zeros_like(input_image)
    for row in range(rows):
        for col in range(columns):
            r,c = np.matmul(translation_matrix,array([row-x_mid,col-y_mid]))
            r1 = floor(r+x_mid)
            r2 = ceil(r+x_mid)
            c1 = floor(c+y_mid)
            c2 = ceil(c+x_mid)
            #if (r>=0 and r<rows) and (c>=0 and c<columns):
                #new_image[row][col] = input_image[r][c]
            if(r1>=0 and r1<rows and r2>=0 and r2<rows and c1>=0 and c1<columns and c2>=0 and c2<columns):
                bottom_left = input_image[r1][c1]
                bottom_right = input_image[r1][c2]
                top_left = input_image[r2][c1]
                top_right = input_image[r2][c2]
                new_image[row][col] = bottom_right*(r%1)*(1-(c%1)) + bottom_left*(1-r%1)*(1-c%1)+top_right*(r%1)*(c%1) + top_left*(1-r%1)*(c%1)

    fig,axes = plt.subplots(1,2, constrained_layout=True, gridspec_kw={'width_ratios':[1,1]})
    axes[0].imshow(input_image,cmap)
    axes[0].axis("on")
    
    im = axes[1].imshow(new_image,cmap)
    axes[1].axis("on")
    
    cbar = fig.colorbar(im,ax=axes.ravel().tolist())
    #cbar.ax.set_yticklabels([0,255])
    plt.show()

input_file = "../data/barbaraSmall.png"
input_image = mpimg.imread(input_file,format="png")*255.0
myImageRotation(input_image,30)
