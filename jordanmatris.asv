% This file contains help functions aimed at making the code more readable

% PtoNmatrix is used to find number of Jordan blocks of a given size

% kernelDims finds the dimensions of ker(A-lambda*I)^k for k less than or
% equal to the multiplicity of lambda

function J = jordanmatris(A,tol)
    
    %our choice of tolerance
    favorittolerans = 1e-3;

    % Sets tolerance if no input given
    if nargin < 2
        tol = favorittolerans; 
    end
    
    % Calculates eigenvalues ev and their corresponding multiplicities 
    [ev,mult] = heltalsev(A,tol);

    % Creates a row vector containing the diagonal elements of J
    diagEls = [];

    for i = 1:size(ev,1)
        for j = 1:mult(i)
            diagEls = [diagEls, ev(i)];
        end
    end
    
    % Initializes J with the correct diagonal elements
    J = diag(diagEls);

    % sets elements diagonal above main diagonal = 1
    % (This diagonal is corrected later)

    J = J + diag(size(diagEls, 1) - 1, 1);
    
    % Calculates a sequence block sizes from the number of blocks of given
    % size for each eigenvalue
    % The nestled loops go through each eigenvalue, and for each eigenvalue
    % calculates the number of blocks of different 
    % sizes in the jordan form.
    % Then for each block size that size is added to blockSqns the number
    % of times it is to appear in the jordan form

    blockSqns = [];

    for i = (1:size(ev,1))
        blockSizes = blockSizes(A, ev(i), mult(i));
        for j = (1:mult(i))
            if(blockSizes(j) ~= 0)
                for k = (1:blockSizes(j))
                    blockSqns = [blockSqns, j];
                end
            end
        end
    end

    % Using blockSqns, we remove 1:s on the off-diagonal where there
    % arent supposed to be any.
    rowCounter = 0;
    for i = (1 : size(blockSqns,1) - 1)
        rowCounter = rowCounter + blockSqns(i);
        J(rowCounter, rowCounter + 1) = 0;
    end

    
end


% PtoNmatrix creates 
% the matrix NtoPmat mapping the numbers n_k of blocks of size k to the
% dimension of the kernel of (A-lambda*I), where lambda is an 
% eigenvalue of A and inverts it to the matrix PtoNmatrix:

% See theorem 7.9 in Holst, Ufnarovski for clarification and proof of
% invertibility

% params eigenMult - row and column size for desired PtoNmatrix
function M = PtoNmatrix(eigenMult)

    NtoPmat = zeros(eigenMult);

    for i = 1:eigenMult
        for j = 1:eigenMult
            NtoPmat(i,j) = min(i,j); 
        end
    end 
    M = inv(NtoPmat);

end

%kernelDims calculates the dimensions of kernels of (A-lam*I)^k for
% k less than or equal to eigenMult, and returns them in a column vector

% params A - the matrix A
% params lam - eigenvalue to calculate dim(ker(A-lam*I)^k) for
% params eigenMult - largest order of k for which dim(ker(A-lam*I)^k) is
% calculated for

function KD = kernelDims(A, lam, eigenMult)
    KD  = zeros(eigenMult,1);
    for i = (1:eigenMult)
        kernelbasis = null((A-eye(size(A))*lam)^i);
        basissize = size(kernelbasis);
        kerDim = basissize(2);
        KD(:,i) = kerDim;
    end
end


% BlockSizes takes a matrix, its eigenvalues and their multiplicities and
% returns the number of blocks of each size less than or equal to eigenMult
% in a row vector 

% params A - the matrix A
% params lam - eigenvalue to calculate dim(ker(A-lam*I)^k) for
% params eigenMult - largest order of k for which dim(ker(A-lam*I)^k) is
% calculated for
function BS = blockSizes(A, lam, eigenMult)
    kerDims  = kernelDims(A, lam, eigenMult); 
    PtoN = PtoNmatrix(eigenMult);
    BS = transpose(PtoN*kerDims);
end

