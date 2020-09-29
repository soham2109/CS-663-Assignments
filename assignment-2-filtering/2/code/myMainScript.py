from myBilateralFiltering import myBilateralFiltering
from time import time

start = time()
files = ["../data/barbara.mat","../data/grass.png","../data/honeyCombReal.png"]

for file_name in files:
    if "barbara" in file_name:
        combinations = [(2.25,25),(2.5,22.5),(2.5,25),(2.5,27.5),(2.75,25)]
    elif "grass" in file_name:
        combinations = [(1.35,25),(1.5,22.5),(1.65,25),(1.5,27.5),(1.5,25)]
    elif "honey" in file_name:
        combinations = [(1.35,25),(1.5,22.5),(1.65,25),(1.5,27.5),(1.5,25)]
    for sigma_s,sigma_int in combinations:
        myBilateralFiltering(file_name,sigma_s,sigma_int)
            
end = time()
print("Total time taken : ",(end-start)/60,"minutes")

