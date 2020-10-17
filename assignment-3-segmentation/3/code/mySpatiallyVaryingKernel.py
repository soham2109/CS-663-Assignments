import cv2
import numpy as np
import sys,os
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import sqrt,e,power,linspace,outer,pi
def im2double(im,d):
    
    im1 = im[:,:,0]
    im2 = im[:,:,1]
    im3 = im[:,:,2]
    out1 = (im1.astype(np.float64) - np.min(im1.ravel())) / (np.max(im1.ravel()) - np.min(im1.ravel()))
    out2 = (im2.astype(np.float64) - np.min(im2.ravel())) / (np.max(im2.ravel()) - np.min(im2.ravel()))
    out3 = (im3.astype(np.float64) - np.min(im3.ravel())) / (np.max(im3.ravel()) - np.min(im3.ravel()))
    out = cv2.merge((out1,out2,out3))
    return out

def meanShift(img):
    r,c,d = img.shape
    img = im2double(img,d)
    gaussian_blur = cv2.GaussianBlur(img, (5,5), 1.0)
    row,col = 128,128
    newimg = cv2.resize(img,(row,col))

    
    result1 = np.zeros((row,col))
    result2 = np.zeros((row,col))
    result3 = np.zeros((row,col))
    h=0.1
    sigma = 11.0 
    count = 0
    window_size = 7
    padded = np.concatenate((np.concatenate((np.zeros((row,window_size,3)), newimg),axis=1), np.zeros((row,window_size,3))),axis=1)
    padded = np.concatenate((np.concatenate((np.zeros((window_size,col+2*(window_size),3)),padded),axis=0), np.zeros((window_size, col+2*(window_size),3))),axis=0)
    r,c,d = padded.shape
    print(padded.shape)
    spatial = np.zeros((2*window_size+1,2*window_size+1))
    for is1 in range(2*window_size+1):
        for is2 in range(2*window_size+1):
            spatial[is1][is2] = ((is1-window_size)**2+(is2-window_size)**2)**0.5
    spatial = np.exp(-(spatial/sigma)**2)
   
    for idx1 in tqdm(range(window_size,r-window_size)):
        for idx2 in range(window_size,c-window_size):
            window = padded[idx1-window_size:idx1+window_size+1, idx2-window_size:idx2+window_size+1]
            #print(newimg[idx1][idx2])
            (x1,x2,x3) = newimg[idx1-window_size][idx2-window_size]
            N1 = 0.0 #numerator
            D1 = 1.0 #denominator
            N2 = 0.0 #numerator
            D2 = 1.0 #denominator
            N3 = 0.0 #numerator
            D3 = 1.0 #denominator


            for itern in range(30): # number of iterations
                for idx3 in range(2*window_size+1):
                    for idx4 in range(2*window_size+1):
                        (x_i1,x_i2,x_i3) = window[idx3][idx4]
                        diff1 = abs(x1-x_i1)
                        diff2 = abs(x2-x_i2)
                        diff3 = abs(x3-x_i3)

                        d1 = np.exp(-(diff1/h)**2)*spatial[idx3][idx4]
                        n1 = x_i1*d1
                        N1 += n1
                        D1 += d1

                        d2 = np.exp(-(diff2/h)**2)*spatial[idx3][idx4]
                        n2 = x_i2*d2
                        N2 += n2
                        D2 += d2

                        d3 = np.exp(-(diff3/h)**2)*spatial[idx3][idx4]
                        n3 = x_i3*d3
                        N3 += n3
                        D3 += d3

                x1 = float(N1)/D1 # in each iteration, x changes
                x2 = float(N2)/D2 # in each iteration, x changes
                x3 = float(N3)/D3 # in each iteration, x changes

            result1[idx1-window_size][idx2-window_size] = x1
            result2[idx1-window_size][idx2-window_size] = x2
            result3[idx1-window_size][idx2-window_size] = x3
    result = cv2.merge((result1,result2,result3))
    return result,newimg

def dnorm(x,mu,sigma):
    return 1 / (sqrt(2 * pi) * sigma) * e ** (-power((x - mu) / sigma, 2) / 2)

def gaussian_kernel(ksize,mu=0,sigma=1,verbose=False):
    # create the 1-D gaussian kernel
    if ksize%2==0:
        ksize = ksize+1
    
    kernel_1D = linspace(-(ksize // 2), ksize // 2, ksize)
    for i in range(ksize):
        kernel_1D[i] = dnorm(kernel_1D[i], mu, sigma)
    kernel_2D = outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / np.sum(kernel_2D)
    return kernel_2D



def convolution(image, kernel_distances):

    image_row, image_col,dim = image.shape
    print(image_row,image_col) 
        
    output = np.zeros_like(image)
    
    plt.figure()
    plt.imshow(image/np.max(image))
    
    padded_row,padded_col,_ = image.shape
    print(image.shape)
    
    for row in tqdm(range(padded_row)):
        for col in range(padded_col):
            
            min_truncate = min(min(row,padded_row-row-1),min(col,padded_col-col-1))
            kernel_size = min(min_truncate,kern_size(kernel_distances[row,col,0]))
            
            if kernel_size!=0:
                kernel = gaussian_kernel(kernel_size+2,sigma=kern_size(kernel_distances[row,col,0]))
                #print(row,col,kernel.shape,row+kernel.shape[0])
                for i in range(3):
                    output[row,col,i] = np.sum(kernel *image[row-kernel.shape[0]//2:row+kernel.shape[0]//2+1, col-kernel.shape[1]//2:col + kernel.shape[1]//2+1,i])
            else:
                for i in range(3):
                    output[row,col,i] = image[row,col,i]
                                         
            
    return output

def mask(result2,resized_orig,threshold,distance_thresh):

    masked = np.zeros_like(result2)
    unique_intensities ={}
    result = (result2).astype(np.float32)
    hsv_result = cv2.cvtColor(result,cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv_result)
    r,c = v.shape
    diff = np.zeros_like(v)

    for i in tqdm(range(r)):
        for j in range(c):
            inten = v[i,j]
            flag=0
            if inten not in unique_intensities:
                diff = 999999
                intensity = 0
                for k in unique_intensities:
                    if abs(inten-k)<distance_thresh:
                        flag=1
                        if diff>abs(inten-k):
                            diff = abs(inten-k)
                            intensity=k 
                        v[i,j] = intensity
                if flag==0:
                    unique_intensities[inten] = 0
                else:
                    unique_intensities[intensity] += 1
            else:
                unique_intensities[inten] += 1

    sort =sorted(unique_intensities.items(), key = lambda kv:(kv[1], kv[0])) 
    print(sort)
    h_copy = np.ones_like(h)
    s_copy = np.ones_like(s)
    v_copy = v.copy()
    for i in tqdm(range(r)):
        for j in range(c):
            if v_copy[i,j]==sort[-1][0]:
                v_copy[i,j] = 0
            else:
                v_copy[i,j] = 1
    masked = cv2.merge((h,s,v))
    masked_2 = cv2.merge((h,s,v_copy))
    masked = cv2.cvtColor(masked,cv2.COLOR_HSV2RGB)
    masked_2 = cv2.cvtColor(masked_2,cv2.COLOR_HSV2RGB)

    vectorized_mask1,img1,foreground_pixels1=masked_image(masked_2,resized_orig,0,threshold)
    vectorized_mask2,img2,foreground_pixels2=masked_image(masked_2,resized_orig,1,threshold)

    return vectorized_mask1,vectorized_mask2,img1,img2,foreground_pixels1,foreground_pixels2

def masked_image(masked_2,resized_orig,flag,threshold):
    r,c,d = masked_2.shape
    vectorized_mask = np.zeros((r,c))
    foreground_pixels = []
    for i in range(r):
        for j in range(c):
                vectorized_mask[i,j] = masked_2[i,j,0] * masked_2[i,j,1] * masked_2[i,j,2]
                if vectorized_mask[i,j]>threshold:   #0.02 works for flower , 0.4 for bird
                    if flag==0:
                        vectorized_mask[i,j]=0
                    else:
                        vectorized_mask[i,j]=1
                    foreground_pixels.append((i,j))
                else:
                    if flag==0:
                        vectorized_mask[i,j]=1
                    else:
                        vectorized_mask[i,j]=0   

   
    masked_img = np.zeros((result2.shape[0],result2.shape[1],3))
    masked_img[:,:,0] = resized_orig[:,:,0]*vectorized_mask
    masked_img[:,:,1] = resized_orig[:,:,1]*vectorized_mask
    masked_img[:,:,2] = resized_orig[:,:,2]*vectorized_mask
    masked_img = masked_img/np.max(masked_img)
    return vectorized_mask,masked_img,foreground_pixels

def plot_save(resized_image,vectorized_mask1,masked_img1,masked_img2,filename):
    fig,axes = plt.subplots(2,2, constrained_layout=True)
    axes[0][0].imshow(resized_image)
    axes[0][0].axis("on")
    axes[0][0].set_title("Original Image")
    axes[0][1].imshow(vectorized_mask1,cmap='gray')
    axes[0][1].axis("on")
    axes[0][1].set_title("Mask Image")
    #cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    axes[1][0].imshow(masked_img1)
    axes[1][0].axis("on")
    axes[1][0].set_title("Masked Image Foreground 0")
    im = axes[1][1].imshow(masked_img2)
    axes[1][1].axis("on")
    axes[1][1].set_title("Masked Image Foreground 1")
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    
    plt.savefig("../images/"+filename+"Mask_blur.png",bbox_inches="tight",pad=-1)







if __name__=="__main__" : 

    filename = sys.argv[1]
    threshold = float(sys.argv[2])
    alpha = float(sys.argv[3])
    distance_thresh = float(sys.argv[4])

    img = cv2.imread(filename) # Read image here
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result2,resized_image = meanShift(img)
    filename = filename.split("/")[-1].split(".")[0]

    vectorized_mask1,vectorized_mask2,masked_img1,masked_img2,foreground_pixels1,foreground_pixels2 = mask(result2,resized_image,threshold,distance_thresh)  #0.4 for bird, 0.02 for flower
    
    plot_save(resized_image,vectorized_mask1,masked_img1,masked_img2,filename+"_"+str(alpha)+"_")
    #name = filename.rstrip("\n").split("/")[-1].split(".")[0]
    
    r,c,d = masked_img1.shape

    masked_image_distance = np.zeros_like(masked_img1)
    masked_image_kern_sizes = np.ones_like(masked_img1)*5

    kern_size = lambda sigma: int(round(((sigma-0.8)/0.3+1)*2+1 )) if sigma >0.8 else 0

    for i in tqdm(range(r)):
        for j in range(c):
            distance_min = 9999999
            for k in foreground_pixels1:
                if ((i-k[0])**2+(j-k[1])**2)<distance_min**2:
                    distance_min = np.sqrt((i-k[0])**2+(j-k[1])**2)
                    masked_image_distance[i,j,:] = distance_min
                    
            if distance_min>=alpha:
                masked_image_distance[i,j,:] = alpha


    gaussian_blurred = convolution(resized_image,masked_image_distance)
    
    fig,axes = plt.subplots(1,2, constrained_layout=True)
    axes[0].imshow(resized_image)
    axes[0].axis("on")
    axes[0].set_title("Original Image")
    im = axes[1].imshow(gaussian_blurred)
    axes[1].axis("on")
    axes[1].set_title("Blurred Image")
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.45)
    
    plt.savefig("../images/"+filename+"_"+str(alpha)+"_"+"background_blur.png",bbox_inches="tight",pad=-1)

    plt.figure()
    plt.imshow(masked_image_distance[:,:,0],cmap='jet')
    plt.axis("on")
    plt.title("Variation with distance")
    plt.colorbar()
    plt.savefig("../images/"+filename+"_"+str(alpha)+"_"+"distance.png",bbox_inches="tight",pad=-1)


