function [Z, D, iter, obj_values, z_con_values] = tssrc(X, lambda, beta, mu, rho)

% default parameters
% rho = 6;
max_mu = 1e6;
tol = 1e-3;
max_iter = 50;

n = size(X, 2);
Z = zeros(n, n);
Y = zeros(n, n);

D = X;
% Dtmp = D;

z_con_values = zeros(1, max_iter);
obj_values = zeros(1, max_iter);


iter = 0;
obj_tmp = 0;
while iter < max_iter;
    
    % update J    
    tmp1 = lambda * (D' * D) + mu * eye(n);
    tmp2 = lambda * (D' * X) + mu * Z - Y;   
    J = normc(tmp1 \ tmp2);
        
    % update Z
    tmp = J + Y/mu;
    thr = sqrt(lambda / mu);
    Z = tmp.*((sign(abs(tmp)-thr)+1)/2);
    ind = abs(abs(tmp)-thr) <= 1e-6;
    Z(ind) = 0;
    Z = Z - diag(diag(Z));
    
    % update D
    A = Z * Z' + beta / lambda * eye(n); 
    B = X * Z';
    for i = 1 : n
        if(A(i, i) ~= 0)
            a = 1.0 / A(i,i) * (B(:,i) - D * A(:, i)) + D(:,i);
            D(:,i) = a / (max( norm(a, 2),1));		
        end
    end
                    
     % update Lagrange multiplier
    Y = Y + mu * (J - Z);
    
    % update penalty parameter
    mu = min(rho * mu, max_mu);
      
%     if(iter > 1)
%         diff_value = abs(last_ratio - ratio);
%         if (diff_value < 1e-6)
% %             disp(diff_value);
%         end
%     end
%     last_ratio = ratio;
    
    err = max(max(abs(J - Z)));
    iter = iter + 1; 
    if err < tol
        break;
    end  
    
    z_con_values(iter) = err;
    obj = length(find(abs(Z) > 1e-6)) + lambda *  norm((X -D * Z), 'fro') + beta * norm(D, 'fro');
    err = abs(obj - obj_tmp) / abs(obj);
%     disp(err);
    obj_values(iter) = err;
    obj_tmp = obj;   
   
end

end


