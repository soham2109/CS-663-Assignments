import numpy as np

def SVD(m=4,n=4):
    #random_matrix = np.random.randn(m,n)
    random_matrix = np.identity(n)
    U,diag,V_T = np.linalg.svd(random_matrix)
    print("U svd:",U)
    print("Diag svd:",diag)
    print("V.T svd:",V_T)
    print("********************")
    print(random_matrix)

    temp1 = np.matmul(random_matrix,random_matrix.T)
    
    lambda_,U = np.linalg.eig(temp1)
    #U=U.T
    print("lambda: ",lambda_) 
    sorted_lambda = np.sort(lambda_)[::-1]
    print("Lambda sorted :   ",sorted_lambda)
    
    idx = np.argsort(lambda_)[::-1]
    print("idx",idx)
    
    U_sort = np.zeros_like(U)

    print("U: ",U)
    _,c = U_sort.shape
    for i in range(c):
        U_sort[:,i] = U[:,idx[i]]

    print("U sorted: ",U_sort)
    
    #idx = np.argsort(lambda_)
    #U_sort = np.take_along_axis(U,idx,axis=0)
    
    temp2 = np.matmul(random_matrix.T,random_matrix)
    
    lambda_,V = np.linalg.eig(temp2)
    #V=V.T
    print("lambda: ",lambda_) 
    sorted_lambda = np.sort(lambda_)[::-1]
    idx_ = np.argsort(lambda_)[::-1]
    print("idx",idx_)
    V_sort = np.zeros_like(V)
    
    print("V: ",V)
    _,c = V_sort.shape
    for i in range(c):
        V_sort[:,i] = V[:,idx_[i]]

    print("Lambda sorted :",sorted_lambda)
    print("V_sort_T: ",V_sort.T)
    
    print("*****************************************")
    print(np.matmul(np.matmul(U_sort,np.diag(np.sqrt(sorted_lambda))),V_sort.T))
    


if __name__=="__main__":
    SVD()
