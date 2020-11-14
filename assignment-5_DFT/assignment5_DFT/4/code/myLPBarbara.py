import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_image(fname, verbose=True, is_rgb=False):
    """Read the image and return as an array
    :param fname : the path to the input image
    :param verbose : if True saves the read image
    :param is_rgb : if true means the read image is RGB
    :output image : the image as a numpy array
    """
    params = {"fname": fname.split("/")[-1].split(".")[0], "attrib": "Original Image"}
    if is_rgb:
        image = cv2.imread(fname)
    else:
        image = cv2.imread(fname,0)
    if verbose:
        save_images(image, params)
    return image

def distance(point1, point2):
    """Returns euclidean distance between 2 2D points
    :param point1, point2 : the two 2D-points under consideration
    :output : the euclidean distance
    """
    return np.sqrt(np.sum(np.square(point1-point2)))

def idealFilterLP(D0, r, c, verbose= True):
    """Create an ideal Low-Pass filter with a given cutoff frequency and filter shape
    :param D0 : the cut-off frequency of the low pass filter
    :param r,c : the shape of the filter (r: rows, c: cols)
    :param verbose : if True saves the filter constructed as an image
    :output base : the constructed filter
    """
    params={"filter": "IdealLP", "D0": D0, "r": r, "c": c, "attrib": "Ideal Low Pass Filter with D0 : "+str(D0)}
    base = np.zeros((r,c))
    center = np.array([r//2,c//2])
    for x in range(c):
        for y in range(r):
            if distance(np.array([y,x]),center) < D0:
                base[y,x] = 1
    
    if verbose:
        save_images(base, params)

    return base

def gaussianLP(D0, r, c, verbose=True):
    """Create a Gaussian LowPass filter with a given cut-off frequency and filter size
    :param D0 : the cut-off frequency for the gaussian LP filter
    :param r,c : the shape of the Gaussian Filter
    :param verbose : if True saves the filter constructed as an image
    :output base : the constructed Gaussian Filter
    """
    params={"filter": "GaussianLP","D0": D0, "r": r, "c": c,"attrib": "Gaussian Low Pass Filter with D0 :"+str(D0)}
    base = np.zeros((r,c))
    center = np.array([r//2,c//2])
    for x in range(c):
        for y in range(r):
            base[y,x] = np.exp(((-distance(np.array([y,x]),center)**2)/(2*(D0**2))))
    
    if verbose:
        save_images(base, params)

    return base

def multiply(fname, image, filterKernel, ktype, D ,verbose=True):
    """Filters the image in frequency domain and returns in spatial domain
    :param image: the input image in spatial domain
    :param filterKernel : the LP filter kernel in frequency domain
    :output inverse_img : the filtered image in spatial domain
    """
    name = fname.split("/")[-1].split(".")[0]
    fft_image = np.fft.fftshift(np.fft.fft2(image)) 
    mult_full = fft_image * filterKernel
    r,c = mult_full.shape
    #print("Shape Before : ",mult_full.shape)
    center = (r//2, c//2)
    mult = mult_full[center[0] - r//4:center[0] + r//4, center[1]-c//4: center[1]+c//4]
    #print("Shape After : ",mult.shape)
    inverse_img = np.abs(np.fft.ifft2(np.fft.ifftshift(mult)))

    if verbose:
        save_images(np.log(1+np.abs(fft_image)), {"attrib": "FFT Image", "fname": name,"type": "FFT_Image","kernel_tyoe": ktype, "D0": str(D) })
        save_images(np.log(1+np.abs(mult)), {"attrib": "Fourier Filtered Image", "fname": name, "type": "Fourier_Filtered","kernel_tyoe": ktype, "D0": str(D) })
        save_images(inverse_img, {"attrib": "Filtered Image","type": "Reconstructed", "fname": name, "kernel_type": ktype, "DO": str(D)})

    return inverse_img

def save_images(image, params, is_rgb=False):
    """Saves the images based on the parameters passed
    :param image : the image to be saved
    :param params : the parameters of the image on construction
    """
    keys = params.keys()
    path = "../images/"
    filename = ""
    title = ""
    for key in params:
        if key!="attrib":
            filename += str(params[key]) + "_"

    for key in params:
        if key in ["fname","attrib"]:
            title += str(params[key]) + "_"

    filename = path + filename + ".png"
    plt.figure()
    plt.title(title)
    if is_rgb:
        plt.imshow(image)
        plt.colorbar()
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.savefig(filename,cmap="gray", bbox_inches="tight")

def pad_image(image):
    """Padd the input image to twice its size
    :param image: input image
    :output out : the double uniformly padded image
    """
    r,c = image.shape[:2]
    out = np.zeros((2*r, 2*c))
    for i in range(r):
        for j in range(c):
            out[2*i,2*j] = image[i,j]

    return out

def main(fname,D,types):
    kern = {"idealLP": idealFilterLP, "gaussLP": gaussianLP}
    in_image = read_image(fname)
    image = pad_image(in_image)
    r,c = image.shape[:2]
    for t in types:
        for d in D:
            kernel = kern[t](d,r,c)
            out = multiply(fname, image, kernel, t, d )
            
if __name__=="__main__":
    filename = "../data/barbara256.png"
    D = [40,80]
    types=["idealLP","gaussLP"]
    main(filename, D, types)
