
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numpy import zeros,zeros_like,array,c_
from math import cos,sin,pi,floor,ceil

def myImageRotation(input_file,angle,cmap="gray"):
    input_image = mpimg.imread(input_file,format="png")
    name = input_file.split(".")[2].split("/")[2]
    
    theta = angle * pi/180
    translation_matrix = array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    
    rows = input_image.shape[0]
    columns = input_image.shape[1]
    new_image = np.ones_like(input_image)
    x_mid = (rows -1 )/2
    y_mid = (columns -1 )/2
    
    for row in range(rows):
        for col in range(columns):
            # rotation matrix transform
            [row_prime,col_prime] = np.matmul(translation_matrix,array([row-x_mid,col-y_mid]).T)
            r=x_mid+row_prime
            r1 = floor(r)
            r2 = ceil(r)
            c = y_mid+col_prime
            c1 = floor(c)
            c2 = ceil(c)
    
            if(r1<rows and r2<rows and c1<columns and c2<columns and r1>=0 and r2>=0 and c1>=0 and c2>=0):
                bottom_left = input_image[r1][c1]
                bottom_right = input_image[r2][c1]
                top_left = input_image[r1][c2]
                top_right = input_image[r2][c2]
                new_image[row][col] = bottom_right*(r%1)*(1-(c%1)) + bottom_left*(1-r%1)*(1-c%1)+top_right*(r%1)*(c%1) + top_left*(1-r%1)*(c%1)

    fig,axes = plt.subplots(1,2, constrained_layout=True)
    axes[0].imshow(input_image,cmap="gray")
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    
    im = axes[1].imshow(new_image,cmap="gray")
    axes[1].axis("on")
    axes[1].set_title(r"Rotated Image by $30^{o}$")

    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    plt.savefig("../images/"+name+"RotateBy"+str(angle)+".png",cmap=cmap,bbox_inches="tight",pad=-1)
    plt.imsave("../images/"+name+"Rotate.png",new_image,cmap=cmap)

    
    
    
