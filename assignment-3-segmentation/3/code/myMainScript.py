from mySpatiallyVaryingKernel import all_functions

filenames = ["bird.jpg","flower.jpg"]
foldername = "../data/"


#for i in range(len(filenames)):
all_functions("../data/flower.jpg",0.02,20.0,0.40)
all_functions("../data/bird.jpg",0.4,5.0,0.35)
