import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from math import ceil,floor
from numpy import zeros,zeros_like,array
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import c_

input_file = "../data/barbaraSmall.png"
input_image = mpimg.imread(input_file,format="png")*255

from myBilinearInterpolation import *
from myNearestNeighbourInterpolation import *
from myBicubicInterpolation import *
from VisualizeDifferent import *

myShrinkImageByFactorD(input_image)
myBilinearInterpolation(input_image)
myNearestNeighbourInterpolation(input_image)
myBicubicInterpolation(input_image)
myImageRotation(input_image)
visualize(input_image)

