import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from tqdm import tqdm
from math import floor
import cv2


def myForegroundMask(input_file,cmap="gray",offset=2):
    name = input_file.split(".")[2].split("/")[2]
    input_image = cv2.imread(input_file,0)
    new_image = np.zeros_like(input_image)
    
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    print(np.max(input_image))
    
    r,c = input_image.shape[:2]
    
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            if input_image[i][j] < np.mean(input_image)+offset:
                new_image[i][j]=0
            else:
                new_image[i][j]=1
    
    masked_image = new_image*input_image
        
    fig,axes = plt.subplots(1,3, constrained_layout=True)
    axes[0].imshow(input_image,cmap="gray")
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    
    axes[1].imshow(new_image,cmap="gray")
    axes[1].axis("on")
    axes[1].set_title("Foreground Mask(th = 2)")
    
    im = axes[2].imshow(masked_image*2,cmap="gray")
    axes[2].axis("on")
    axes[2].set_title("Masked Image")
    
    
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)    
    plt.savefig("../images/"+name+"ForegroundMask.png",cmap=cmap,bbox_inches="tight",pad=-1)
    
    plt.imsave("../images/" + name+"Mask.png",new_image,cmap=cmap)
    cv2.imwrite("../images/" + name+"ForegroundMasked.png",masked_image)
    
