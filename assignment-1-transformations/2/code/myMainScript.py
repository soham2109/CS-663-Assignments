from myHM import myHM
from myHE import myHE
from myForegroundMask import myForegroundMask
from myCLAHE import myCLAHE
from myLinearContrastStretching import myLinearContrastStretching

from time import time

image1 = "../data/barbara.png"
image2 = "../data/TEM.png"
image3 = "../data/canyon.png"
image4 = "../data/retina.png"
image5 = "../data/church.png"
image6 = "../data/chestXray.png"
image7 = "../data/statue.png"

maskedStatue = "../data/statueForegroundMasked.png"
reference = "../data/retinaRef.png"
reference_mask = "../data/retinaRefMask.png"
target = "../data/retina.png"
target_mask = "../data/retinaMask.png"

superStart = time()

start = time()
myForegroundMask(image7)
end = time()
print("Time to run myForegroundMask.py :",end-start,"secs")

start = time()
myLinearContrastStretching(image1,[120,230],[90,245])
myLinearContrastStretching(image2,[120,230],[30,245])
myLinearContrastStretching(image3,[100,180],[60,200])
myLinearContrastStretching(image5,[10,200],[180,230])
myLinearContrastStretching(image6,[100,200],[30,180])
myLinearContrastStretching(maskedStatue,[50,200],[20,230])
end = time()
print("Time to run myLinearContrastStretching.py :",end-start,"secs")


start = time()
myHE(image1)
myHE(image2)
myHE(image3)
myHE(image5)
myHE(image6)
myHE(maskedStatue)
end = time()
print("Time to run myHE.py :",end-start,"secs")

start = time()
myHM(reference,reference_mask,target,target_mask)
end = time()
print("Time to run myHM.py :",end-start,"secs")

start = time()
myCLAHE(image1,64,64,0.005,"gray")
myCLAHE(image1,128,128,0.005,"gray")
myCLAHE(image1,16,16,0.005,"gray")
myCLAHE(image1,64,64,0.0025,"gray")
myCLAHE(image2,32,32,0.03,"gray")
myCLAHE(image2,64,64,0.03,"gray")
myCLAHE(image2,4,4,0.03,"gray")
myCLAHE(image2,32,32,0.0015,"gray")
myCLAHE(image3,16,16,0.005,"gray")
myCLAHE(image3,32,32,0.005,"gray")
myCLAHE(image3,8,8,0.005,"gray")
myCLAHE(image3,16,16,0.0025,"gray")
myCLAHE(image6,8,8,0.015,"gray")
myCLAHE(image6,32,32,0.015,"gray")
myCLAHE(image6,2,2,0.015,"gray")
myCLAHE(image6,8,8,0.0075,"gray")
myCLAHE(maskedStatue)
end = time()
print("Time to run myCLAHE.py :",end-start,"secs")

superEnd = time()
print("Time required to run codes for Q2 :",round(superEnd-superStart,2),"minutes")



