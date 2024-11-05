import numpy as np
from scipy.linalg import eig, null_space

def jordanMatrix(A, tol= None):
    ev, mult = heltalsEV(A, tol)
    diagEls = []

    if(mult[0] == -1):
        return "Non integer eigenvalues"

    for i in range(0, len(ev)):
        for j in range(0, mult[i]):
            diagEls.append(ev[i])

    J = np.diag(diagEls)

    for k in range(1,len(diagEls)):
        J[k-1][k] = 1
    
    blockSqns = []
    for i in range(0, len(ev)):
        blockSize = blockSizes(A, ev[i], mult[i])
        for j in range(0, mult[i]):
            if(blockSize[j]!=0):
                for k in range(0, int(blockSize[j])):
                    blockSqns.append(j+1)
    
    rowcounter = 0
    for i in range(0, len(blockSqns)-1):
        rowcounter += blockSqns[i]
        J[rowcounter-1][rowcounter] = 0
    return np.real(J)


def blockSizes(A, eigV, mult):
    kerDim = np.transpose(kernelDim(A, eigV, mult))
    ptoMat = ptoNMatrix(mult)
    BS = np.matmul(ptoMat, kerDim)
    return BS

def kernelDim(A, eigV, mult):
    KD = np.zeros(mult)

    for i in range(0, mult):
        b = np.linalg.matrix_power(A - np.identity(len(A))*eigV, i+1)
        rank = np.linalg.matrix_rank(b)
        KD[i] = len(A) - rank
    return KD

def ptoNMatrix(mult):
    ntopmat = np.zeros((mult, mult))

    for i in range(0,mult):
        for j in range(0, mult):
            ntopmat[i, j] = np.min([i+1, j+1])
    return np.linalg.inv(ntopmat)


def heltalsEV(A, tol = None):
    favTol = 1e-7

    if(tol == None):
        tol = favTol
    
    eigenValues = np.linalg.eigvals(A)
    integer_eigenValues = np.round(eigenValues)
    not_tolerated = any(np.abs(eigenValues-integer_eigenValues) > tol)
    not_tolerated_complex = any(np.abs(np.imag(eigenValues)) > tol)
 
    if(not_tolerated or not_tolerated_complex):
        return [-1], [-1]
    else: 
        ev, mult = np.unique(integer_eigenValues, return_counts=True)
        return ev, mult



from numpy.polynomial.polynomial import Polynomial
def test_matrix(n):
    """
    Generates a specific test matrix based on the input integer n.
    These matrices are meant to be used as test cases for Jordan form calculations.
    
    Parameters:
    n (int): An integer specifying which matrix to generate.

    Returns:
    numpy.ndarray: The generated matrix.
    
    Raises:
    ValueError: If n is not an integer between 0 and 6.
    """
    if n == 0:
        # Zero matrix 3x3
        M = np.zeros((3, 3))
    elif n == 1:
        # Identity matrix 4x4
        M = np.eye(4)
    elif n == 2:
        # Matrix with repeating rows
        M = np.array([[1, 1, 1], [1, 1, 1], [-2, -2, -2]])
    elif n == 3:
        # Matrix with all eigenvalues equal to zero
        # MATLAB may struggle with this due to numeric instability
        M = np.array([
            [-9, 11, -21, 63, -252],
            [70, -69, 141, -421, 1684],
            [-575, 575, -1149, 3451, -13801],
            [3891, -3891, 7782, -23345, 93365],
            [1024, -1024, 2048, -6144, 24572]
        ])
    elif n == 4:
        # Companion matrix of a polynomial with roots 1 through 10
        roots = np.arange(1, 11)
        p = Polynomial.fromroots(roots).coef[::-1]  # Get polynomial coefficients
        M = np.diag(np.ones(len(roots) - 1), k=-1)  # Create companion matrix structure
        M[0, :] = -p[1:] / p[0]  # Fill first row with normalized coefficients
    elif n == 5:
        # Complex eigenvalues case
        M = np.array([[3, -4], [4, 3]])
    elif n == 6:
        # Matrix with nearly identical eigenvalues for tolerance testing
        M = np.diag([1.000001, 1])
    else:
        raise ValueError("The argument should be an integer between 0 and 6.")

    return M

M = test_matrix()

print(jordanMatrix(M))
