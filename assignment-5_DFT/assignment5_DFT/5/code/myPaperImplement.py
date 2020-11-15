import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import cv2


def create_rectangle(size, start , shape, verbose=True):
    I = np.zeros((size, size))
    for i in range(start[0],start[0]+shape[0]+1):
        for j in range(start[1], start[1]+shape[1]+1):
            I[i,j] = 255

    if verbose:
        plt.figure()
        plt.title("Original Rectangle")
        plt.set_cmap("gray")
        plt.imshow(I, cmap="gray")
        plt.colorbar()
        plt.savefig("../images/OriginalRectangle.png", bbox_inches="tight")
    return I

def spatial_translation(image, translation, verbose=True):
    translation_matrix = np.array([[1,0, translation[0]],[0,1, translation[1]]]).astype(np.float)
    translated = cv2.warpAffine(image, translation_matrix, image.shape )
    
    # translation_simple = np.zeros_like(image)
    # tx, ty = translation
    # r,c = image.shape
    # for i in range(r):
    #     for j in range(c):
    #         translation_simple[i,j] = image[(i+tx)%r, (j+ty)%c]


    if verbose:
        plt.figure()
        plt.title("Original Translated Rectangle")
        plt.set_cmap("gray")
        plt.imshow(translated, cmap="gray")
        plt.colorbar()
        plt.savefig("../images/OriginalTranslatedRectangle.png", bbox_inches="tight")
       
    #   plt.figure()
    #   plt.imshow(translation_simple, cmap="gray")
    #   plt.show()
    return translated

def calculate_Fourier(image):
    return np.fft.fftshift(np.fft.fft2(image))

def cross_power_spectrum(orig, translated, verbose=True):
    
    orig_fourier = calculate_Fourier(orig)
    trans_fourier = calculate_Fourier(translated)
    cross_power = orig_fourier * np.conj(trans_fourier)
    cross_power = cross_power/np.abs(orig_fourier*trans_fourier)
    inv_cross = np.fft.ifftshift(np.fft.ifft2(cross_power))
    mag_inv_cross = np.abs(inv_cross)
    print(mag_inv_cross[mag_inv_cross>0])
    
    if verbose:
        plt.figure()
        plt.imshow(mag_inv_cross)
        plt.show()

if __name__=="__main__":
    size = 300
    start = (50,50)
    shape = (50,70)
    orig_rect = create_rectangle(size, start, shape)
    orig_trans = spatial_translation(orig_rect, (-30, 70))
    cross_power_spectrum(orig_rect, orig_trans)

