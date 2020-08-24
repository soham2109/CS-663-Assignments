import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

from myBilinearInterpolation import *
from myNearestNeighbourInterpolation import *
from myBicubicInterpolation import *

def visualize(input_image):
    #input_file = "../data/barbaraSmall.png"
    #input_image = mpimg.imread(input_file,format="png")*255
    
    region = input_image[50:80,50:80]
    
    myBilinearInterpolation(region,cmap="jet")
    myNearestNeighbourInterpolation(region,cmap="jet")
    myBicubicInterpolation(region,cmap="jet")

input_file = "../data/barbaraSmall.png"
input_image = mpimg.imread(input_file,format="png")*255
visualize(input_image)
