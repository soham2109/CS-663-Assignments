import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import cv2


def create_rectangle(size, start , shape, type_, scale=1.0, verbose=True):
    I = np.zeros((size, size))
    for i in range(start[0],start[0]+shape[0]+1):
        for j in range(start[1], start[1]+shape[1]+1):
            if type_=="Original":
                I[i,j] = 255
            elif type_=="Noisy":
                I[i,j] = 255 + np.random.normal(scale=20.0)

    if verbose:
        plt.figure()
        plt.title(type_+" Rectangle")
        plt.set_cmap("gray")
        plt.imshow(normalize(I), cmap="gray")
        plt.colorbar()
        plt.savefig("../images/"+type_+"_Rectangle.png", bbox_inches="tight")
    return I

def spatial_translation(image, translation, type_, verbose=True):
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
    return np.fft.fft2(image)

def cross_power_spectrum(orig, translated, type_, verbose=True):
    
    orig_fourier = calculate_Fourier(orig)
    trans_fourier = calculate_Fourier(translated)
    cross_power = np.fft.ifft2(orig_fourier*np.conj(trans_fourier))
    eps = 1e-15
    ir = np.abs(np.fft.ifft2((orig_fourier * trans_fourier.conjugate()) / (np.abs(orig_fourier) * np.abs(trans_fourier)+eps)))
    
    if verbose:
        plt.figure()
        plt.imshow(np.log(1+ir),cmap="gray")
        plt.colorbar()
        plt.title(type_ +" Log Cross-Power Spectrum")
        plt.savefig("../images/LogCrossPowerSpectrum_"+type_ +".png",bbox_inches="tight",cmap="gray")
    
    r,c = orig.shape
    t0, t1 = np.unravel_index(np.argmax(ir), orig.shape)
    if t0 >r//2:
        t0 -= r
    if t1>c//2:
        t1 -= c
    return [t1, t0]

def normalize(image):
    max_ = np.max(image)
    min_ = np.min(image)

    return (image-min_)/(max_-min_)

if __name__=="__main__":
    size = 300
    start = (50,50)
    shape = (50,70)
    std_dev = 20
    
    orig_rect = normalize(create_rectangle(size, start, shape, "Original"))
    noisy_rect = normalize(create_rectangle(size, start, shape, "Noisy", scale=20.0))

    orig_trans = normalize(spatial_translation(orig_rect, (-30, 70),"Original"))
    noisy_trans = normalize(spatial_translation(noisy_rect, (-30, 70), "Noisy"))
    
    t0,t1 = cross_power_spectrum(orig_rect, orig_trans, "Original")
    print("tx: {}, ty: {} for Original Rectangle.".format(t0, t1))
    
    t0,t1 = cross_power_spectrum(noisy_rect, noisy_trans, "Noisy")
    print("tx: {}, ty: {} for Noisy Rectangle.".format(t0, t1))

