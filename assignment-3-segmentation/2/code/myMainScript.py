
from myMeanShiftSegmentation import meanShift

filenames = ["baboonColor.png","bird.jpg","flower.jpg"]
foldername = "../data/"
iterations = [30,40,40]

for i in range(len(filenames)):
    meanShift(foldername+filenames[i],intensity_sigma=0.1,num_iter=iterations[i])
