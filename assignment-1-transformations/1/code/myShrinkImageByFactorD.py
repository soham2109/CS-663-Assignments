import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def myShrinkImageByFactorD(d,cmap="gray"):
    """
    d : List of 
    Shrink the image size by a factor of d along each dimension using image subsampling by
    sampling / selecting every d -th pixel along the rows and columns.
    This is the function that takes in the mutliples D to resize the image
    along with a colormap
    """
    input_file = "../data/circles_concentric.png"
    input_image = mpimg.imread(input_file,format="png")
    num_plots = len(d)+1
    output_images = []
    
    width,height = input_image.shape
    count = 0

    for i in d:
        count = count + 1
        new_width = int(width/i)
        new_height = int(height/i)
        output = zeros((new_width,new_height))
        for W in range(new_width):
            for H in range(new_height):
                output[W][H] = input_image[W*i][H*i]
        
        output_images.append(output)
    
    parameters = {'axes.titlesize': 10}
    plt.rcParams.update(parameters)

    fig,axes = plt.subplots(1,num_plots, constrained_layout=True)
    

    axes[0].imshow(input_image,cmap=cmap)
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    for i in range(count):
        im = axes[i+1].imshow(output_images[i], cmap=cmap)
        axes[i+1].axis("on")
        axes[i+1].set_title("Shrinked by factor d="+str(d[i]))

    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.35)
    plt.savefig("../data/circleShrinked.png",bbox_inches="tight",pad=-1)
    
    for i in output_images:
        plt.imsave("../data/Shrinkage"+str(d[output_images.index(i)]) +".png",i,cmap=cmap)
    
    
myShrinkImageByFactorD([2,3])
