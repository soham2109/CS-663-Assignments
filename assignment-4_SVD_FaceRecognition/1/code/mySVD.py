import numpy as np

def SVD(matrix):
    """Calculates SVD using eigen value decomposition method and 
        returns the U,S,V_T
    :param matrix: any random matrix
    :output U,S,V_T : the singular value components
    """
    # calculating A'A
    temp1 = np.matmul(matrix.T,matrix)
    # calculating eigen values and vectors of A'A
    lambda_,V = np.linalg.eig(temp1)
 
    # sorting the singular values and in descending order
    sorted_lambda = np.sort(lambda_)[::-1]
    
    idx = np.argsort(lambda_)[::-1]
    V_sort = np.zeros_like(V)
    _,c = V_sort.shape
    
    for i in range(c):
        V_sort[:,i] = V[:,idx[i]]

    U_sort = np.matmul(matrix,V_sort)
    _,c = U_sort.shape
    for i in range(c):
        U_sort[:,i] = U_sort[:,i]/np.sqrt(np.sum(np.square(U_sort[:,i])))
    
    sigma = np.sqrt(sorted_lambda)
    return U_sort,sigma,V_sort.T


if __name__=="__main__":
    m,n = 4,4
    matrix = np.random.randint(1,10,(m,n))
    print("Input matrix of size {}x{}:\n {}".format(m,n,matrix))
    print("****************************************")
    U,S,V_T = SVD(matrix)
    print("U matrix\n",U)
    print("Singular Values\n",S)
    print("V'(V transpose) Matrix\n",V_T)
    print("*****************************************")
    print("calculated matrix: \n",np.matmul(np.matmul(U,np.diag(S)),V_T))
