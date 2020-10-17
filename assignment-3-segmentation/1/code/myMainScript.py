from myHarrisCornerDetection import HarrisCornerDetector

filename="../data/boat.mat"
window_size = 7
k = 0.06
thresh = 0.8*10**(-5)
sigma_weights = 1.2
print("Window Size: " + str(window_size))
print("K-value for Cornerness measure: " + str(k))
print("Corner Response Threshold:" + str(thresh))
HarrisCornerDetector(filename, int(window_size),sigma_weights, float(k), thresh)
