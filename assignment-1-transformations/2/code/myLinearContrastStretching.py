import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from numpy import zeros_like
from tqdm import tqdm
import cv2
from seaborn import distplot

def plot_hist(input_file,input_image,output_image):
    """
    input : input_file_path, input_image, output_image
    output : saves the histograms for both the images for comparison
    dependencies : seaborn, numpy, matplotlib
    """
    name = input_file.split(".")[2].split("/")[2]

    
    ax = distplot(input_image,color='r',label ="Input Histogram",hist_kws={"alpha": 0.3, "linewidth": 1.5},bins=256,hist=False)
    ax = distplot(output_image,color="b",label ="Contrast Stretched Histogram",hist_kws={"alpha": 0.3,"linewidth": 1.5},bins=256,hist=False)
    l1 = ax.lines[0]
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    ax.fill_between(x1,y1, color="red", alpha=0.3)
    l2 = ax.lines[1]
    x2 = l2.get_xydata()[:,0]
    y2 = l2.get_xydata()[:,1]
    ax.fill_between(x2,y2, color="blue", alpha=0.3)
    plt.title("Normalized Histogram Plots for Images")
    plt.legend()
    plt.savefig("../images/"+name+"LCSHistogram.png",bbox_inches="tight",pad=-1)
    

def truncate(array):
    """
    input : array
    output : truncated array to make it stay from 0 to 255
    """
    r,c = array.shape
    for i in range(r):
        for j in range(c):
            if array[i][j]<0.0:
                array[i][j] = 0
            elif array[i][j]>255.0:
                array[i][j]=255.0
    return array
        

def myLinearContrastStretching(input_file,x1=[0,255],x2=[0,255],cmap="gray"):
    """
    input : <input_file_path>, input_image_range(x1), output_image_range(x2), cmap(optional)
    output : Saves the linear contrast stretched image
    x1 : [r1,r2] (by default = [0,255])
    x2 : [s1,s2] (by default = [0,255])
    """
    
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    r1,r2 = x1
    s1,s2 = x2
    name = input_file.split(".")[2].split("/")[2]
    input_image = cv2.imread(input_file)
    d = 1
    if len(input_image.shape)>2:
        r,c,d = input_image.shape
    else:
        r,c = input_image.shape
    
    if d==1:
        new_image=np.zeros_like(input_image)
        for i in tqdm(range(r)):
            for j in range(c):
                input_pixel = input_image[i,j]
                if input_pixel<=r1:
                    input_pixel = (s1/r1) * input_pixel
                elif input_pixel>r1 and input_pixel <= r2:
                    input_pixel = ((s2-s1)/(r2-r1))*(input_pixel-r1) + s1
                else:
                    input_pixel = ((255.0-s2)/(255.0-r2))*(input_pixel-r2) + s2
                new_image[i][j]= input_pixel
        new_image = truncate(new_image)
        plot_hist(input_file,input_image,new_image)
        
    else:
        input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(input_image,cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(hsv_image)
        v_new = v.copy()
        for i in tqdm(range(r)):
            for j in range(c):
                input_pixel = v[i,j]
                if input_pixel<=r1:
                    input_pixel = (s1/r1) * input_pixel
                elif input_pixel>r1 and input_pixel <= r2:
                    input_pixel = ((s2-s1)/(r2-r1))*(input_pixel-r1) + s1
                else:
                    input_pixel = ((255.0-s2)/(255.0-r2))*(input_pixel-r2) + s2
                v_new[i,j]= input_pixel
                
        hsv_image[:,:,2] = v_new
        plot_hist(input_file,v,v_new)
        new_image = cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)
        

    fig,axes = plt.subplots(1,2, constrained_layout=True)
    axes[0].imshow(input_image,cmap="gray")
    axes[0].axis("on")
    axes[0].set_title(r"Original Image")
    im = axes[1].imshow(new_image,cmap="gray")
    axes[1].axis("on")
    axes[1].set_title(r"Linear Contrast Stretched Image")
    
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    plt.savefig("../images/"+name+"LCS.png",bbox_inches="tight",pad=-1)
    
    plt.imsave("../images/" + name+"LinearContrastStretching.png",new_image,cmap=cmap)
