filenames = ["baboonColor.png","bird.jpg","flower.jpg"]
foldername = "../data/"
iterations = [30,40,40]


from myMeanShift import meanShift

for i in range(len(filenames)):
    meanShift(foldername+filenames[i],intensity_sigma=3.0,num_iter=iterations[i])
