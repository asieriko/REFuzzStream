function [Z, iter, errors] = online_dic_sample_selection(X, beta)
% Inputs:
%   X: each coloum of X represents a sample (m*n).
%   beta: a parameter
% Outputs:
%  Z: n*n
% The author: Jie Chen, chenjie2010@scu.edu.cn

tol = 5e-3;
max_iter = 500;
errors = zeros(1, max_iter);

m = size(X, 2);
Z_tmp = zeros(m, m);
sigma = eye(m);
iter = 1;
XX = X' * X;
while iter <= max_iter
    
    % update Z
%     Z = inv(1 / beta * sigma + XX) * XX;
    Z = (1 / beta * sigma + XX) \ XX;
    Z = Z - diag(diag(Z));
    
    % update sigma
    for i = 1 : m
        sigma(i, i) = 1 / (norm(Z(i, :), 2) + eps);
    end

    err = max(max(abs(Z_tmp - Z)));
    errors(1, iter) = err;
    iter = iter + 1; 
    if err < tol 
%         disp([iter, err]);
        break;
    end
    Z_tmp =  Z;    
%     disp([iter, err]);

end
