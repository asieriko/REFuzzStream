function [X, ground_labels] = generate_data(num_points, num_subspace, dim_subspace, dim_space, percentage_noise)

% n = 200;
% d = 10;
% D = 100;

    [U, ~, ~] = svd(rand(dim_space));
    ground_labels = [];
    U1 = U(:, 1 : dim_subspace);
    bases = [];
    bases = [bases U1];
    X = U1 * rand(dim_subspace, num_points);
    ground_labels = [ground_labels, ones(1, num_points)];
    
    for i = 2 : num_subspace
        R = orth(rand(dim_space));
        U1 = R * U1;
        bases = [bases U1];
        X = [X, U1 * rand(dim_subspace, num_points)];
        ground_labels = [ground_labels, i * ones(1, num_points)];
    end
    
    if percentage_noise > 0
        nX = size(X, 2);
        norm_x = sqrt(sum(X.^2,1));
        norm_x = repmat(norm_x, dim_space, 1);
        gn = norm_x .* randn(dim_space, nX);
        inds = rand(1, nX) <= percentage_noise;
        X(:,inds) = X(:,inds) + gn(:, inds);
    end

end