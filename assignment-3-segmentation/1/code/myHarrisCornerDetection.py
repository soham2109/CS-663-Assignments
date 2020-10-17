import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

def normalize_image(image):
    out = image.copy()
    normalized = (out-np.min(out))/(np.max(out)-np.min(out))
    return normalized

def get_image(filename):
    f = h5py.File(filename,"r")
    out = f.get('imageOrig')
    out = np.array(out)
    return normalize_image(out)

def dnorm(x,sigma,mu=0):
    return (1.0/(np.sqrt(2*np.pi)*sigma))*np.e**(-(((x - mu)/sigma)**2) / 2)

def gaussian_kernel(ksize,sigma):
    
    kernel_1D = np.linspace(-(ksize // 2), ksize // 2, ksize)
    for i in range(ksize):
        kernel_1D[i] = dnorm(kernel_1D[i], sigma=sigma)

    # computers outer product of two 1-D gaussian kernels
    # to produce a 2D Gaussian Kernel
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0/np.sum(kernel_2D)
    return kernel_2D


def convolution(image,kernel_size,sigma):
    kernel = gaussian_kernel(kernel_size,sigma=sigma)
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)

    # CREATING ZERO-PADDED IMAGE
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    # CONVOLUTION OPERATION DONE HERE
    for row in tqdm(range(image_row),desc="Gaussian Convolution"):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    output = (output/np.max(output)) 

    return output

def calculate_gradient(image):
    dy, dx = np.gradient(image)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    return (dy,dx,Ixx,Iyy,Ixy)

def cornernessMeasure(Sxx,Syy,Sxy,k):
    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    r = det - k*(trace**2)
    return r,det,trace

def findCorners(filename, window_size_blur, sigma_weights, k, thresh):
    """
    Finds and returns list of corners and new image with corners drawn
    :param img: The original image
    :param window_size: The size (side length) of the sliding window
    :param k: Harris corner constant. Usually 0.04 - 0.06
    :param thresh: The threshold above which a corner is counted
    :return:
    """
    
    img = get_image(filename)
    sigma_window = 0.3*((window_size_blur-1)*0.5 - 1) + 0.8
    img = convolution(img,window_size_blur,sigma_window)
    #img = img*255.0
    img = np.rot90(img)
    
    #Find x and y derivatives
    (dy,dx,Ixx,Iyy,Ixy) = calculate_gradient(img) 
    height,width = img.shape
    
    plt.figure()
    plt.imshow(dy,cmap='inferno',origin="lower")
    plt.title("Derivative along Y")
    plt.colorbar()
    plt.savefig('../images/y_derivative.png',cmap='inferno',bbox_inches="tight")
    
    plt.figure()
    plt.imshow(dx,cmap='inferno',origin="lower")
    plt.title("Derivative along X")
    plt.colorbar()
    plt.savefig('../images/x_derivative.png',cmap='inferno',bbox_inches="tight")
    
    cornerList = []
    offset = int(window_size_blur/2)
    
    eigvalues = np.zeros((height,width,2))
    corner_img =np.zeros((height,width))
    
    #Loop through image and find our corners
    min_r = 1000000
    max_r = 0
    for y in tqdm(range(offset, height-offset),desc="Finding Corners..."):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            weight_kernel = gaussian_kernel(2*offset+1,sigma=sigma_weights)
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]*weight_kernel
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]*weight_kernel
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]*weight_kernel
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            r,det,trace = cornernessMeasure(Sxx,Syy,Sxy,k)
            eigvalues[x,y,0] = (trace + np.sqrt(trace**2 - 4*det))/2
            eigvalues[x,y,1] = (trace - np.sqrt(trace**2 - 4*det))/2
            corner_img[x,y]=r
            
            if(r>max_r):
                max_r = r
            if (r<min_r):
                min_r = r
            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
                cornerList.append([x, y, r])
    
    print("Minimum Cornerness Value :",min_r)
    print("Maximum Cornerness Value :",max_r)
    return img, cornerList,corner_img ,eigvalues

def plotCorners(finalImg,cornerList):
    plt.figure()
    plt.imshow(finalImg,cmap="gray",origin="lower")
    plt.colorbar()
    plt.title("Red Corners in Image")
    for i in cornerList:
        k,j,l = i
        plt.scatter(k,j,color="r",s=0.3,marker="*")
    plt.savefig('../images/Harris.png',bbox_inches='tight')
    
def plotEigenValues(eigvalues):
    plt.figure()
    plt.imshow(np.rot90(eigvalues[:,:,0]),cmap='inferno')
    plt.title("Eigen Value 1")
    plt.colorbar()
    plt.savefig('../images/Eigen_value1.png',cmap='inferno',bbox_inches='tight')
    plt.figure()
    plt.imshow(np.rot90(eigvalues[:,:,1]),cmap='inferno')
    plt.title("Eigen Value 2")
    plt.colorbar()
    plt.savefig('../images/Eigen_value2.png',cmap='inferno',bbox_inches='tight')
    
def plotCornernessMeasure(corner_img):
    plt.figure()
    plt.imshow(np.rot90(corner_img),cmap='hot')
    plt.title("Cornerness Measure Plot")
    plt.colorbar()
    #plt.show()
    plt.savefig('../images/Cornerness.png',cmap='hot')
    
def HarrisCornerDetector(filename,window_size_blur, sigma_weights, k, thresh):
    img, cornerList,corner_img ,eigvalues = findCorners(filename,window_size_blur, sigma_weights, k, thresh)
    plotCorners(img,cornerList)
    plotEigenValues(eigvalues)
    plotCornernessMeasure(corner_img)
    


if __name__=="__main__":
    filename="../data/boat.mat"
    window_size = 7
    k = 0.06
    thresh = 0.8*10**(-5)
    sigma_weights = 1.2
    print("Window Size: " + str(window_size))
    print("K-value for Cornerness measure: " + str(k))
    print("Corner Response Threshold:" + str(thresh))
    HarrisCornerDetector(filename, int(window_size),sigma_weights, float(k), thresh)
