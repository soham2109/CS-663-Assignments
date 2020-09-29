from myUnsharpMasking import unSharpMask

files = ["../data/superMoonCrop.mat","../data/lionCrop.mat"]
for file_name in files:
    if "lion" in file_name:
        alpha = 1.6
        kernel = 15
    elif "superMoon" in file_name:
        alpha = 2.2
        kernel = 19
   
    unSharpMask(file_name,kernel,alpha)
