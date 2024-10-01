function Xs = construct_representative_objects(X, D, Z, n, lables, num_clusters)

    avg_num = floor(n / num_clusters);
    if avg_num == 0
        avg_num = 1;
    end
%     for idx = 1 : num_clusters
%         positions = (lables == idx);
%         len = length(positions);
%         
%     end
%     total_num = sum(num_set);
    Xs = zeros(size(X, 1), n);
    
    next_pos = 1;
    for idx = 1 : num_clusters
        positions = find(lables == idx);
        len = length(positions);        
        if len <= avg_num
            current_num = len;
        else
            current_num = avg_num;
        end        
        errors = zeros(1, len);
%         Xi = D(:, positions) * Z(positions, positions); 
%         XX = normc(X(:, positions));
        for j = 1 : len
            if j == 1
                current_positions = positions(2 : len);
            elseif j == len
                current_positions = positions(1 : len - 1);
            else
                current_positions = positions;
                current_positions(j) = [];
            end
%             errors(j) = norm(XX - D(:, current_positions) * Z(current_positions, positions), 'fro');
            errors(j) = norm(D(:, current_positions) * Z(current_positions, positions), 'fro');
        end    
%         [~, idx_set] = sort(errors, 'descend');
        [~, idx_set] = sort(errors);
        Xs(:, next_pos : next_pos + current_num - 1) = X(:, idx_set(1 : current_num));
        next_pos = next_pos + current_num;            
%         if current_num <= length(idx_set)
%             aa = 0;
%         end
        Xs = Xs(:, 1 : next_pos - 1); 
    end
end

