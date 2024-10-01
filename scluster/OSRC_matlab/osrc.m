function [Z, iter, errors] = osrc(X, lambda, rho)
% Inputs:
%   X: each coloum of X represents a sample (m*n).
%   lambda: a parameter
%   rho: step size (rho > 1)
% Outputs:
%  Z: the similiarity matrix (n*n)
% The author: Jie Chen, chenjie2010@scu.edu.cn

% default parameters
mu = 1e-2;
max_mu = 1e6;
tol = 1e-6;
max_iter = 500;
errors = zeros(1, max_iter);

if nargin < 4
    rho = 1.6;
end

[dim, n] = size(X);
Z = zeros(n, n);
Y = zeros(n, n);
XX = X * X';
D = X;
iter = 1;
while iter <= max_iter
        
    % update J
    DD = D' * D;
    tmp1 = lambda * DD + mu * eye(n);
    tmp2 = lambda * DD + mu * Z - Y;   
    J = normc(tmp1 \ tmp2);
        
    % update Z
    tmp = J + Y/mu;
    thr = sqrt(1 / mu);
    Z = tmp.*((sign(abs(tmp)-thr)+1)/2);
    ind = abs(abs(tmp)-thr) <= 1e-6;
    Z(ind) = 0;
    Z = Z - diag(diag(Z));

    E = X - X * Z;
    EE = E * E';
    
    %update P
    [v, ~] = eig(EE, XX);
    P = v(:, 1 : dim);    
    
    % learning
    D = P' * X;
    
     % update Lagrange multiplier
    Y = Y + mu * (J - Z);
    
    % update penalty parameter
    mu = min(rho * mu, max_mu);
    
    err = max(max(abs(J - Z)));
    errors(1, iter) = err;
    iter = iter + 1; 
    if err < tol
        break;
    end  
    
end
