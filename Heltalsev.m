%Function heltalsev 
% params A - Matrix input
% params tol - Tolerance 

function [ev,mult] = heltalsev(A,tol) 

    %Our choice for tolerance
    favorittolerans = 1e-3;
    
    %if number of arguments is less than 2 we set the tol to
    %favorittolerans
    if nargin < 2
        tol = favorittolerans; 
    end
    
    % calculates eigenvalues
    eigenvalues = eig(A);
    %integer part
    integer_eigenvalues = round(eigenvalues);
    %Checks if it is within tolerance
    is_tolerated = abs(eigenvalues-integer_eigenvalues)<tol;
    
    %if tolerated, we calculate the unique eigenvalues and 
    % the multiplicity of the eigenvalues 
    if all(is_tolerated)
        ev = unique(integer_eigenvalues);
        mult = histcounts(integer_eigenvalues, ev);
    else
        % We return empty arrays 
        ev = [];
        mult = [];
        fprintf('Över Tolerans\n');
    end




% [ev, mult] = heltalsev(M)
