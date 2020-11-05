import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

def find_mean(array):
    mean = np.mean(array,axis=0)
    
    return mean

def im2double(im):
    min_val = np.min(im)
    max_val = np.max(im)
    out = (im.astype('float') - min_val) / (max_val - min_val)
    
    return out

def read_train_images_ORL(file_path,subjects,images_per_subject):
    image_array = []
    for i in range(1,subjects+1):
        folder_path = os.path.join(file_path,"s"+str(i))
        #print(folder_path)
        for j in range(1,images_per_subject+1):
            filepath = os.path.join(folder_path,str(j)+".pgm")
            im = cv2.imread(filepath,0)
            image_array.append(im2double(im).ravel())
            
    return np.array(image_array)

def read_test_images_ORL(file_path,subjects,images_per_subject,starting_index):
    image_array = []
    for i in range(1,subjects+1):
        folder_path = os.path.join(file_path,"s"+str(i))
        for j in range(starting_index,starting_index+images_per_subject):
            filepath = os.path.join(folder_path,str(j)+".pgm")
            im = cv2.imread(filepath,0)
            image_array.append(im2double(im).ravel())
            
    return np.array(image_array)

def subtract_mean(mean,train,test):
    train_mean = train-mean
    test_mean = test-mean
    return train_mean, test_mean

def normalize_vecs(array):
    _,c = array.shape
    for i in range(c):
        array[:,i] = array[:,i]/(np.sqrt(np.sum(np.square(array[:,i]))))
    return array


def calculate_svd(array):
    pass

def calculate_evd(array):
    L_train = np.matmul(array,array.T)
    eig_val,W = np.linalg.eig(L_train)
    V = np.matmul(array.T,W)
    V = normalize_vecs(V)
    return V
    
def calculate_alpha(V,train,test):
    alpha_train = np.matmul(V.T,train.T)
    alpha_test = np.matmul(V.T,test.T)
    return alpha_train, alpha_test

def calculate_and_plot_prediction_rates(train,test,alpha_train,alpha_test,ks,dataset,light=False):
    prediction_rate = []
    for k in ks:
        correct_prediction_count = 0
        for counter, ele in enumerate(alpha_test.T):
            index = np.argmin(np.sum(np.square((alpha_train.T[:,:k]-alpha_test.T[counter,:k])),axis=1))
            if (counter//test == index//train):
                correct_prediction_count += 1
        prediction_rate.append(correct_prediction_count/float(alpha_test.shape[1]))
    
    plt.figure()
    plt.plot(ks,prediction_rate,"bo")
    plt.plot(ks,prediction_rate,alpha=0.7,linestyle='dashed')
    plt.ylabel('Prediction Rate')
    plt.xlabel('Values of k')
    plt.ylim(ymin=0,ymax=1)
    plt.title("Prediction rate in {} dataset vs k".format(dataset))
    plt.grid()
    if not light:
        plt.savefig("../images/PredRate_"+dataset+"_normal.png")
    else:
        plt.savefig("../images/PredRate_"+dataset+"_withoutLighting.png")

        

def ORL_data(path, subjects, train_images_per_subject, test_images_per_subject, ks):
    train_images = read_train_images_ORL(path,subjects,train_images_per_subject)
    test_images = read_test_images_ORL(path,subjects,test_images_per_subject,starting_index=train_images_per_subject+1)
    train_mean = find_mean(train_images)
    train_mean_subtracted, test_mean_subtracted = subtract_mean(train_mean, train_images, test_images)
    eigen_vectors = calculate_evd(train_mean_subtracted)
    alpha_train, alpha_test = calculate_alpha(eigen_vectors, train_mean_subtracted, test_mean_subtracted)
    calculate_and_plot_prediction_rates(train_images_per_subject,test_images_per_subject,alpha_train, alpha_test, ks,dataset="ORL")

    
if __name__=="__main__":
    ORL_PATH = "../../ORL"
    ORL_k = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170]
    ORL_data(ORL_PATH,subjects=32,train_images_per_subject=6,test_images_per_subject=4,ks=ORL_k)
    
    