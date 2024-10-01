function [A, k, sse, sc] = spectral_clustering_with_max_k(Z, k_max, enable_k, wnd_idx, current_size)


k_min = 1;
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

A = kmeans(V, k, 'maxiter', 1000, 'replicates', 20, 'EmptyAction', 'singleton');
% A = litekmeans(V, k, 'MaxIter',100, 'Replicates', 1000);

sse = 0;
sc = 0;
if wnd_idx > 1
    A = A(1 : current_size);        
end
for i = 1 : k
   pos = A == i;
    if ~isempty(pos)
        cluster = V(pos, :);

        %sse
        centroid = mean(cluster, 1);
        clu_size = size(cluster, 1);
        for j = 1 : clu_size
            sse = sse + sum((cluster(j, :) - centroid).^2);
        end

        %sc        
        for j = 1 : clu_size
            x = cluster(j, :);
            sum_x = 0;
            for ii = 1 : clu_size
                if j ~= ii
                    sum_x = sum_x + sqrt(sum((x - cluster(ii, :)).^2));
                end
            end
            a = sum_x / current_size;
            
            idx = 0;
            b_set = zeros(1, k-1);
            for m = 1 : k                
                if i ~= m
                    idx = idx + 1;
                    sum_x = 0;
                    new_pos = A == m;
                    if ~isempty(new_pos)
                        another_cluster = V(new_pos, :);
                        another_clu_size = size(another_cluster, 1);
                        for jj = 1 : another_clu_size
                            sum_x = sum_x + sqrt(sum((x - another_cluster(jj, :)).^2));
                        end
                        b_set(1, idx) = sum_x;
                    end
                end
            end         
            b_set(b_set<1e-6)=[];
            b = min(b_set);
            if a > 1e-6
                sc = sc + (b-a)/max(a,b);
            end
        end
    end
end
sc = sc / current_size;
