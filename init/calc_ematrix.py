import numpy as np

def calc_ematrix(F, K):
    E = K.T @ F @ K

    #(U, S, Vt) = np.linalg.svd(E)

    #Sp = np.eye(3,3)
    #Sp[-1, -1] = 0

    #return U * Sp * Vt

    return E