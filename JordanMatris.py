import numpy as np
from scipy.linalg import eig, null_space

# The code below computes the Jordan form for a given matrix with integer eigenvalues

# We make use of three help functions: ptoNMatrix, kernelDim, blockSizes.
# These are used to increase the readablility of the jordanMatrix function

# The jordanmatrix function works by creating a matrix with the correct main diagonal and ones on the off-diagonal, and then removing 
# ones "north east" of the corner between where one jordan block ends and another begins


# Computes the Jordan normal form for a matrix with integer eigenvalues
# Inputs: A square matrix A, and optionally a tolerance tol
# Returns: The Jordan normal form of the matrix A if A has integer eigenvalues, and None if A has non-integer eigenvalues

def jordanMatrix(A, tol= None):
    ev, mult = heltalsEV(A, tol) # Checks for and computes integer eigenvalues and their multiplicities
    

    if(mult[0] == -1): 
        return None
    
    # Computes a vector containing the diagonal elements in the jordan normal form
    diagEls = [] 

    for i in range(0, len(ev)):
        for j in range(0, mult[i]):
            diagEls.append(ev[i])

    # Initiates a matrix J with the correct main diagonal and ones on the off-diagonal

    J = np.diag(diagEls)

    for k in range(1,len(diagEls)):
        J[k-1][k] = 1
    
    # Calculates the sequence of block sizes in the Jordan form of A
    blockSqns = []
    for i in range(0, len(ev)):
        blockSize = blockSizes(A, ev[i], mult[i]) # computes the number of blocks with size < mult[i] for the eigenvalue ev[i]
        for j in range(0, mult[i]):
            if(blockSize[j]!=0):
                for k in range(0, int(blockSize[j])):
                    blockSqns.append(j+1) 

    # Removes ones on the off-diagonal "northeast" of the corner between blocks, where they were incorrectly inserted previously
    
    rowcounter = 0
    for i in range(0, len(blockSqns)-1):
        rowcounter += blockSqns[i]
        J[rowcounter-1][rowcounter] = 0

    return np.real(J)


# Computes the number of blocks of sizes less than or equal to mult for a given eigenvalue eigV of the matrix A
# Input: A square matrix A, an eigenvalue eigV of A, and the multiplicity mult of said eigenvalue
# Returns: An array containing the number of blocks of size i+1 at position i.

def blockSizes(A, eigV, mult):
    kerDim = np.transpose(kernelDim(A, eigV, mult))
    ptoMat = ptoNMatrix(mult)
    BS = np.matmul(ptoMat, kerDim)
    return BS

# Computes the dimensions p_i = dim( ker(A - eigV*I)^i ) for i less than or equal to mult
# Input: A square matrix A, an eigenvalue eigV of A, and the multiplicity mult of said eigenvalue
# returns: An array containing the p_i:s

def kernelDim(A, eigV, mult):
    KD = np.zeros(mult)

    for i in range(0, mult):
        b = np.linalg.matrix_power(A - np.identity(len(A))*eigV, i+1)
        rank = np.linalg.matrix_rank(b)
        KD[i] = len(A) - rank
    return KD


# Computes the matrix NtoPmat mapping the numbers n_k of blocks of size k to the
# dimension of the kernel of (A-lambda*I), where lambda is an 
# eigenvalue of A and inverts it to the matrix PtoNmatrix.

# See theorem 7.9 in Holst, Ufnarovski for clarification and proof of
# invertibility

# Input: The maximal size mult of a block (or equivalently the multiplicity of an eigenvalue)
# Returns: The matrix mapping dimensions of kernels (A - eigV*I)^i to the number of blocks of a given size

def ptoNMatrix(mult):
    ntopmat = np.zeros((mult, mult))

    for i in range(0,mult):
        for j in range(0, mult):
            ntopmat[i, j] = np.min([i+1, j+1])
    return np.linalg.inv(ntopmat)


# Checks whether the eigenvalues of a given matrix A are sufficiently close to integers, and computes these and their multiplicities¨
# Input: A matrix A and a tolerance tol 
# Returns: None if the eigenvalues are non-integer, and the eigenvalues ev, and their multiplicities otherwise.

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



# These are test cases:

""" from numpy.polynomial.polynomial import Polynomial
def test_matrix(n):
    
    #Generates a specific test matrix based on the input integer n.
    #These matrices are meant to be used as test cases for Jordan form calculations.
    
    #Parameters:
    #n (int): An integer specifying which matrix to generate.

    #Returns:
    #numpy.ndarray: The generated matrix.
    
    #Raises:
    #ValueError: If n is not an integer between 0 and 6.
    
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

M = test_matrix(1)

print(jordanMatrix(M)) """


