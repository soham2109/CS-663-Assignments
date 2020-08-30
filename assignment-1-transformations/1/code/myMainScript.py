from myShrinkImageByFactorD import *
from myNearestNeighbourInterpolation import *
from myBilinearInterpolation import *
from myBicubicInterpolation import *
from myImageRotation import *

from time import time

circle_file = "../data/circles_concentric.png"
D = [2,3]
barbara_file = "../data/barbaraSmall.png"
angle=30

super_start = time()

start = time()
myShrinkImageByFactorD(circle_file,D,cmap="gray")
end = time()
print("Time taken to run myShrinkageImageByFactorD.py :",end-start,"secs")

start = time()
myBilinearInterpolation(barbara_file,cmap="gray")
end = time()
print("Time taken to run myBilinearInterpolation.py :",end-start,"secs")

start = time()
myNearestNeighbourInterpolation(barbara_file,cmap="gray")
end = time()
print("Time taken to run myNearestNeighbourInterpolation.py :",end-start,"secs")

start = time()
myBicubicInterpolation(barbara_file,cmap="gray")
end = time()
print("Time taken to run myBicubicInterpolation.py :",end-start,"secs")

# region test
start = time()
region = [50,80,50,80]
myBilinearInterpolation(barbara_file,region=region,cmap="jet")
myNearestNeighbourInterpolation(barbara_file,region=region,cmap="jet")
myBicubicInterpolation(barbara_file,region=region,cmap="jet")
end = time()
print("Time taken to run region tests :",end-start,"secs")

start = time()
myImageRotation(barbara_file,angle,cmap="gray")
end = time()
print("Time taken to run myImageRotation.py :",end-start,"secs")

super_end = time()
print("Time taken to run all the scripts :",super_end-super_start,"secs")
