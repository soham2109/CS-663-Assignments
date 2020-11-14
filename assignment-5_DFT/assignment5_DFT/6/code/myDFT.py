import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import axes3d

def computeDFT(N, matrix,KernelType, verbose=True):
    center = (N+1)/2
    fac = 2*np.pi/N
    X = np.linspace(-(N-1)/2, (N-1)/2, num=N)
    Y = np.linspace(-(N-1)/2, (N-1)/2, num=N)
    
    my_ft_lap = np.zeros((N,N)).astype(np.complex64)
    for u in range(N):
        for v in range(N):
            if KernelType=="Normal":
                my_ft_lap[u,v] = 2*cmath.exp(-1j*fac*center*(u+v))*(np.cos(fac*u)+np.cos(fac*v)-2)
            elif KernelType.startswith("Diagonal"):
                my_ft_lap[u,v] = 2*cmath.exp(-1j*fac*center*(u+v))*(4-np.cos(fac*u)-np.cos(fac*v)-np.cos(fac*(u+v))-np.cos(fac*(u-v)))
    
    my_ft_lap = np.fft.fftshift(my_ft_lap)
    magnitude = np.log(np.abs(my_ft_lap)+1)

    if verbose:
        plt.figure()
        # plt.colorbar(cmap="jet")
        plt.title("Log Magnitude plot of kernel "+KernelType)
        plt.imshow(magnitude,extent=[-100,100,-100,100])
        plt.set_cmap("jet")
        plt.colorbar()
        #plt.xlim(xmin = np.min(X), xmax=np.max(X))
        #plt.ylim(ymin = np.min(Y), ymax=np.max(Y))
        plt.savefig("../images/"+ KernelType +"LogMagnitude.png", cmap="jet", bbox_inches="tight")
        
        fig = plt.figure() 
        ax = plt.axes(projection ='3d')  
        plt.set_cmap("jet")
        x,y = np.meshgrid(X,Y)
        surf = ax.plot_surface(x, y, magnitude, cmap="jet")
        fig.colorbar(surf, ax=ax)
        plt.title("Surface Plot of "+KernelType + " Laplacian Kernel")
        #plt.show()
        plt.savefig("../images/"+ KernelType +"SurfacePlot.png", cmap="jet", bbox_inches="tight")

if __name__=="__main__":
    N = 201
    laplacianMatrix1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacianMatrix2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    computeDFT(N, laplacianMatrix1, "Normal")
    computeDFT(N, laplacianMatrix2, "Diagonal Added")
