filenames = ["baboonColor.png","bird.jpg","flower.jpg"]
foldername = "../data/"

from myMeanShift import meanShift

for image in filenames:
    meanShift(foldername+image)
