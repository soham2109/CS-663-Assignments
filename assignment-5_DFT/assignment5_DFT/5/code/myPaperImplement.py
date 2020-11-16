import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import cv2
from mpl_toolkits.mplot3d import axes3d

def create_rectangle(size, start , shape, type_, scale=1.0, verbose=True):
    """Creates a rectangular patch in an image
    :param size: the size of the image
    :param start: the starting position of the rectangle
    :param shape: the shape of the rectangular patch
    :param type_: two choices "Noisy" or "Original"
    :param scale: in case of "Noisy", the standard deviation of the noise
    :param verbose: if True plots the patch in the ../images/ folder
    :output I: the rectangular patch image
    """
    if type_=="Original":
        I = np.zeros((size, size))
    elif type_=="Noisy":
        I = np.random.normal(scale=scale,size=size*size).reshape((size,size))

    for i in range(start[0],start[0]+shape[0]+1):
        for j in range(start[1], start[1]+shape[1]+1):
                I[i,j] += 255

    if verbose:
        plt.figure()
        plt.title(type_+" Rectangle")
        plt.set_cmap("gray")
        plt.imshow(normalize(I), cmap="gray")
        plt.colorbar()
        plt.savefig("../images/"+type_+"_Rectangle.png", bbox_inches="tight")
    return I

def spatial_translation(image, translation, type_, verbose=True):
    """Creates a spatially translated image of the rectangular patch
    :param image: the input image to perform translation
    :param translation: the translation co-ordinates
    :param type_: choose between "Noisy" and "Original"
    :param verbose: if True saves the plots
    :output translated: the translated image
    """
    translation_matrix = np.array([[1,0, translation[0]],[0,1, translation[1]]]).astype(np.float)
    translated = cv2.warpAffine(image, translation_matrix, image.shape )
    
    if verbose:
        plt.figure()
        plt.title(type_ + " Translated Rectangle")
        plt.set_cmap("gray")
        plt.imshow(translated, cmap="gray")
        plt.colorbar()
        plt.savefig("../images/" + type_+ "_TranslatedRectangle"+type_+".png", bbox_inches="tight", cmap="gray")
       
    return translated

def calculate_Fourier(image):
    """Calculates the 2D-fourier transform of the input
    :param image: the 2D input image
    :output: 2D Fourier Transformed image
    """
    return np.fft.fft2(image)

def cross_power_spectrum(orig, translated, type_, verbose=True):
    """Calculates the Cross Power Spectrum of the two images and estimates the translation
    :param orig: the original image
    :param translated: the translated original image
    :param type_: "Noisy" or "Original"
    :param verbose: if True saves the Log-Magnitude of Cross Power Spectrum
    :output (t1,t0): the estimated translation
    """
    orig_fourier = calculate_Fourier(orig)
    trans_fourier = calculate_Fourier(translated)
    eps = 1e-15
    ir = np.abs(np.fft.ifft2((orig_fourier * trans_fourier.conjugate()) / (np.abs(orig_fourier) * np.abs(trans_fourier)+eps)))
    
    if verbose:
        plt.figure()
        plt.imshow(np.log(1+ir),cmap="gray")
        plt.colorbar()
        plt.title(type_ +" Log Cross-Power Spectrum")
        plt.savefig("../images/LogInverseCrossPowerSpectrum_"+type_ +".png",bbox_inches="tight",cmap="gray")
        
    r,c = orig.shape
    t0, t1 = np.unravel_index(np.argmax(ir), orig.shape)
    if t0 >r//2:
        t0 -= r
    if t1>c//2:
        t1 -= c
    return [t1, t0]

def normalize(image):
    """Min-Max normalization
    :param image : input image to normalize
    :output min-max scaled image
    """
    max_ = np.max(image)
    min_ = np.min(image)

    return ((image-min_)/(max_-min_))*255.0

def plot_log_magnitude(image,type_):
    r,c = image.shape
    x = np.array([i for i in range(r)])
    y = np.array([i for i in range(c)])
    X,Y = np.meshgrid(x,y)
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    log_mag = np.log(1+np.abs(fft_image))
   
    fig = plt.figure()
    ax = plt.axes(projection ='3d')  
    plt.set_cmap("inferno")
    surf = ax.plot_surface(X, Y, log_mag, cmap="inferno")
    fig.colorbar(surf, ax=ax)
    plt.title("Log Magnitude surf plot of "+type_ + " Image")
    plt.savefig("../images/LogMagSurfPlotof"+type_+"_image.png",bbox_inches="tight",cmap="inferno")


if __name__=="__main__":
    size = 300
    start = (50,50)
    shape = (50,70)
    std_dev = 20
    
    orig_rect = create_rectangle(size, start, shape, "Original")
    noisy_rect = normalize(create_rectangle(size, start, shape, "Noisy", scale=20.0))
    plot_log_magnitude(orig_rect,"Original")
    plot_log_magnitude(noisy_rect,"Noisy")

    orig_trans = spatial_translation(orig_rect, (-30, 70),"Original")
    noisy_trans = spatial_translation(noisy_rect, (-30, 70), "Noisy")
    plot_log_magnitude(orig_trans,"Original Translated")
    plot_log_magnitude(noisy_trans,"Noisy Translated")

    t0,t1 = cross_power_spectrum(orig_rect, orig_trans, "Original")
    print("tx: {}, ty: {} for Original Rectangle.".format(t0, t1))
    
    t0,t1 = cross_power_spectrum(noisy_rect, noisy_trans, "Noisy")
    print("tx: {}, ty: {} for Noisy Rectangle.".format(t0, t1))

