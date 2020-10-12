import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator
import sys
import getopt
import h5py


def get_image(filename):
    f = h5py.File(filename,"r")
    out = f.get('imageOrig')
    out = array(out)

    return (out*255.0/np.max(out))

def normalize_image(image):
    raise NotImplementedError

def gaussian_kernel(sigma):
    raise NotImplementedError

def dnorm(x,y,sigma):
    raise NotImplementedError

def convolution(image,kernel):
    raise NotImplementedError

def calculate_gradient(image):
    raise NotImplementedError

def cornernessMeasure():
    raise NotImplementedError

def myHarrisCornerDetector(filename,sigma_blur,sigma_window,k_corner_response,corner_threshold):
    raise NotImplementedError


def main():
    args, img_name = getopt.getopt(sys.argv[1:], '', ['sigma_blur=','sigma_window=', 'k_corner_response=', 'corner_threshold='])
    args = dict(args)
    print(args)

    signma_blur = args.get('--sigma_blur')
    sigma_window = args.get('--sigma_window')
    k = args.get('--k_corner_response')
    thresh = args.get('--corner_threshold')

    print("Image Name: " + str(img_name[0]))
    print("Sigma Blur: " + str(sigma_blur))
    print("Sigma Window: " + str(sigma_window))
    print("K Corner Response: " + str(k))
    print("Corner Response Threshold:" + thresh)

    img = get_image(img_name[0])

    raise NotImplementedError

if __name__=="__main__:
    main()
