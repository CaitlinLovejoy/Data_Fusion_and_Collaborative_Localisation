import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import datetime
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power


def checkSol (P1_inv):
    solveFlag = 1
    return solveFlag
    P1 = LA.inv(P1_inv)
    # print("P1 in check sol step", P1)
    P1_eigenvalues, S1 = LA.eig(P1)
    if not np.isrealobj(P1_eigenvalues):
        solveFlag = 0
        # print("UNSOLVABLE CASE, COMPLEX EIGENVALUE")
        return solveFlag 
    elif P1_eigenvalues[0] < 0:
        solveFlag = 0
        # print("UNSOLVABLE CASE, NEGATIVE EIGENVALUE")
        return solveFlag 
    elif P1_eigenvalues[1] < 0:
        solveFlag = 0
        # print("UNSOLVABLE CASE, NEGATIVE EIGENVALUE")
        return solveFlag 
    else:
        return solveFlag

def GammaCapDeriv (P1_inv, P2_inv):

    P1 = LA.inv(P1_inv)
    P2 = LA.inv(P2_inv)
    
    # print("P1 in Gamma", P1)
    # print("P2 in Gamma", P2)
    # Step 1: Calculate Eigenvalue decomposition of P1
    P1_eigenvalues, S1 = LA.eig(P1)
   
    
    D1 = np.diagflat(P1_eigenvalues)
    S1_inv = LA.inv(S1)

    
    # print('S1', S1)
    # print('S1 inv', S1_inv)
    # print('D1', D1)
    # print("P1 Eigenvalues", P1_eigenvalues)
    # print("P1 Gamma values", S1, D1, S1_inv)
    # # Step 1.5: Check for n real solutions
    # compCheckA = np.isrealobj(P1_eigenvalues)
    # for i in range(compCheckA.shape):
    #     if compCheckA[i] == False:
    #         print("UNSOLVABLE AS RETURNED COMPLEX EIGENVALUES")

    # Step 2: Calculate P2 section Di^-1/2 @ Si_inv @ Pj @ Si @ Di^-1/2 = Q_ij
    # print("D1 ^1/2", D1**(0.5))
    # D1invHalf = fractional_matrix_power(D1, 0.5)
    # print("D1 invhalf", D1invHalf)
    D1inv = LA.inv(D1)
  
    # print("D1 inv", D1inv)
    D1invHalf = (D1inv**(0.5))
 
    # print("D1 invhalf", D1invHalf)
    
    Q_ij = D1invHalf @ S1_inv @ P2 @ S1 @ D1invHalf

    # for a in Q_ij:
    #     if not np.isfinite(a).all():
            # print(P1, P2)
            # print('Error: Unsolvable case')
            # print(Q_ij)
            # print(D1)

    # Step 3: Calculate Eigenvalue decomposition of Q_ij
    P2_eigenvalues, S2 = LA.eig(Q_ij)

    D2 = np.diagflat(P2_eigenvalues)

    # Step 4: Compute T
    T = S1 @ (D1**(0.5)) @ S2
    

    # Step 5: Compute D2 -> DGamma
    DGammaVal = np.empty(2)
    for i in range(P2_eigenvalues.size):
        if P2_eigenvalues[i] > 1:
            x = P2_eigenvalues[i]
        else:
            x = 1
        DGammaVal[i] = x
    DGamma = np.diagflat(DGammaVal)

    # Step 6: Compute GammaCap
    GammaCap = T @ DGamma @ T.transpose()
    

    return GammaCap, P2_eigenvalues

def gammaLowDeriv (GammaCap, P2_eigenvalues, P1_inv, P2_inv, c1, c2, zeta):

    # Step 1: calculate eta 
    etaVal = np.empty(2)
    for i in range(P2_eigenvalues.size):
        if np.greater_equal(np.abs(P2_eigenvalues[i] - 1), (10*zeta)):
            etaVal[i] = 0
        else: 
            etaVal[i] = zeta
    eta = np.diagflat(etaVal)
    # print("ETA", eta)

    # Step 2: Calculate gammaLow
    ident = np.identity(2)
    GammaCap_inv = LA.inv(GammaCap)
    gammaWorking1 = P1_inv + P2_inv - (2 * GammaCap_inv) + ((2 * eta) @ ident)
    gammaA = np.asarray(gammaWorking1)
    gammaA_inv = LA.inv(gammaA)
    gammaWorking2 = ((P2_inv - GammaCap_inv + (eta @ ident)) @ c1) + ((P1_inv - GammaCap_inv + (eta @ ident)) @ c2)
    gammaB = np.asarray(gammaWorking2)
    gammaLow = gammaA_inv @ gammaB
    # print("Gamma LOW ",gammaLow)
    
    return gammaLow
