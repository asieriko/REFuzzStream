function [A, k] = spectral_clustering_with_max_k(Z, k_max, enable_k)


k_min = 2;
n = size(Z,1);
D = diag(1./sqrt(sum(Z, 2)+ eps));

W = speye(n) - D * Z * D;
[~, sigma, V] = svd(W);
V = V(:, n - k_max + 1 : n);

if enable_k == 0
    k = k_max;
else    
    sigma = diag(sigma);
    s = sigma(n - k_max : n);
    len = length(s) - 1;
    eigengaps = zeros(len, 1);
    for i = 1 : length(eigengaps)
        eigengaps(i) = s(i) - s(i+1);
    end
    [~, k] = max(eigengaps);
    if k < k_min
        k = k_min;
    end
end

for i = 1 : n
    V(i,:) = V(i,:) ./ norm(V(i,:) + eps);
end

% A = kmeans(V, k, 'maxiter', 1000, 'replicates', 20, 'EmptyAction', 'singleton');
A = litekmeans(V, k, 'MaxIter',100, 'Replicates', 1000);
