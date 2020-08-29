import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def truncate(array):
    r,c = array.shape
    for i in range(r):
        for j in range(c):
            if array[i][j]<0.0:
                array[i][j] = 0
            elif array[i][j]>1.0:
                array[i][j]=1.0
    return array
        

def myLinearContrastStretching(input_file,tune=0.375,cmap="gray"):
    
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)
    
    name = input_file.split(".")[2]
    input_image = mpimg.imread(input_file,format="png")
    d = 1
    
    if len(input_image.shape)>2:
        r,c,d = input_image.shape
    else:
        r,c = input_image.shape
    
    # for grayscale images
    if d==1:
        tune = tune
        minimum,maximum = 0,255
        new_image=np.zeros_like(input_image)
        # min-max contrast stretching
        for i in range (input_image.shape[0]):
            for j in range(input_image.shape[1]):
                input_pixel = input_image[i][j]
                x= (input_pixel - minimum -tune)/(maximum-minimum)
                new_image[i][j]=x*255
        new_image = truncate(new_image)
    
    # for RGB images   
    else:
        minimum,maximum = 0,255
        
        f = 1.016*(1+tune)/(1.016-tune)
        new_image=np.zeros_like(input_image)
        for k in range(d):
            new_image[:,:,k] = truncate((f*(input_image[:,:,k] - 0.5) + 0.5))

    fig,axes = plt.subplots(1,2, constrained_layout=True)
    # original image added to the subplot
    axes[0].imshow(input_image,cmap="gray")
    axes[0].axis("on")
    axes[0].set_title(r"Original Image")
    # linear contrasted image added to the subplot
    im = axes[1].imshow(new_image,cmap="gray")
    axes[1].axis("on")
    axes[1].set_title(r"Linear Contrast Stretched Image")
    # Adding the colorbar
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    plt.savefig(".."+name+"LCS.png",bbox_inches="tight",pad=-1)
    
    # Saving only the contrasted image for better visualization
    plt.imsave(".." + name+"LinearContrastStretching.png",new_image,cmap=cmap)
   


input_files = ["../data/chestXray.png","../data/barbara.png",
               "../data/statueForegroundMasked.png","../data/church.png",
               "../data/canyon.png","../data/TEM.png"]
offset_dict = {0: 0.025,1: 0.20, 2: 0.20, 3: -0.05, 4: 0.10, 5: 0.5}

for i in input_files:
    myLinearContrastStretching(i,offset_dict[input_files.index(i)])