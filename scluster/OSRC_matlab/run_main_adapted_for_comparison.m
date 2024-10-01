close all;
clear;
clc;

addpath('utility');
addpath('/data');

lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1];
betas = [1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1];
% dims = [10, 20, 50];

% 1: NOAA; 2: powersupply; 
% 3: DS1; 4: RBF40K; 5: Bench11k
data_index = 2;
switch data_index
    case 1
        % lambdas = [5e-4];
        % betas = [0.5];
        dims = [8];

        load('NEweather.mat');
        filename = "NOAA";
        osrc_data = inputs;
        osrc_labels = targets;
        [row_dim, total_num] = size(inputs);
        n = 1000;
        num_windows = floor(total_num / n);
        K = max(osrc_labels);
        gnd = osrc_labels;
        max_num_per_cluster = floor(n / K);
    case 2
        % lambdas = [5e-4];
        % betas = [0.5];
        dims = [2];

        load('powersupply.mat');
        filename = "PowerSupply";
        osrc_data = inputs;
        osrc_labels = targets+1; % it starts with 0
        [row_dim, total_num] = size(inputs);
        n = 1000;
        num_windows = floor(total_num / n);
        K = max(osrc_labels);
        gnd = osrc_labels;
        max_num_per_cluster = floor(n / K);

  case 3
        % lambdas = [5e-4];
        % betas = [0.5];
        dims = [2];

        load('ds1.mat');
        filename = "DS1";
        osrc_data = inputs;
        osrc_labels = targets;
        [row_dim, total_num] = size(inputs);
        n = 100;
        num_windows = floor(total_num / n);
        K = max(osrc_labels);
        gnd = osrc_labels;
        max_num_per_cluster = floor(n / K);  

    case 4
        % lambdas = [5e-4];
        % betas = [0.5];
        dims = [2];

        load('r4k.mat');
        filename = "RBF1_40k";
        osrc_data = inputs;
        targets(targets==-1)=4;
        osrc_labels = targets;
        [row_dim, total_num] = size(inputs);
        n = 1000;
        num_windows = floor(total_num / n);
        K = max(osrc_labels);
        gnd = osrc_labels;
        max_num_per_cluster = floor(n / K);  

  case 5
        % lambdas = [5e-4];
        % betas = [0.5];
        dims = [2];

        load('b1k.mat');
        filename = "Bench1_11k";
        osrc_data = inputs;
        targets(targets==-1)=4;
        osrc_labels = targets;
        [row_dim, total_num] = size(inputs);
        n = 1000;
        num_windows = floor(total_num / n);
        K = max(osrc_labels);
        gnd = osrc_labels;
        max_num_per_cluster = floor(n / K);
end

lambda_num = length(lambdas);
beta_num = length(betas);
dim_num = length(dims);

final_clustering_accs = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_nmis = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_purities = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_fmeasures = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_ris = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_aris = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_ratios =  zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_iters = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_costs = zeros(lambda_num, dim_num, beta_num, num_windows);
max_iter = 500;
final_clustering_osrc_errors = zeros(lambda_num, dim_num, beta_num, num_windows, max_iter);
final_clustering_dic_iters = zeros(lambda_num, dim_num, beta_num, num_windows);
final_clustering_dic_errors = zeros(lambda_num, dim_num, beta_num, num_windows, max_iter);
max_iters_errors = zeros(1, max_iter);

final_result = strcat(filename, '_final_result.txt');
final_average_result = strcat(filename, '_final_average_result.txt');
final_average_ARI = strcat(filename, '_final_average_ARI.txt');
final_averages_ARI = 'final_averages_ARI.txt';
writematrix(["average_ari", "lambda", "beta"],final_average_ARI, "Delimiter", 'tab', 'WriteMode', 'append'); 
writematrix(["dataset","average_ari", "lambda", "beta"],final_averages_ARI, "Delimiter", 'tab', 'WriteMode', 'append'); 
% final_result_mat = strcat(filename, '_final_result.mat');
final_average_result_mat = strcat(filename, '_final_average_result.mat');

for lambda_idx = 1 : length(lambdas)
    lambda = lambdas(lambda_idx); 
    for beta_idx = 1 : length(betas)
        beta = betas(beta_idx);
        for dim_idx = 1 : length(dims)
            dim = dims(dim_idx);  
            enable_k = 1; % The number of clusters is automatically determined.  
            for wnd_idx = 1 : num_windows
%                 disp(wnd_idx);               
                % 1. Get data and labels
                start_idx = (wnd_idx - 1) * n + 1;
                X = osrc_data(:, start_idx : start_idx + n - 1);                                     
                if wnd_idx == 1                        
                    X_full = X;                        
                else
                    X_full = [X, Xs];
                end
                ground_lables = osrc_labels(start_idx : start_idx + n - 1); 
                
                % 2. Clustering
                %data preprocessing
                if dim > 0
                    [eigen_vector, ~, ~] = f_pca(X_full, dim);
                    XX = eigen_vector' * X_full;
                else
                    XX = X_full;
                end
                tic;
                [Zc, iter, errors] = osrc(normc(XX), lambda);
                time_cost = toc;
                len_samples = size(XX, 2); 
                ratio = length(find(abs(Zc) > 1e-2)) / (len_samples * len_samples);
                Z = abs(Zc) + abs(Zc');
                stream = RandStream.getGlobalStream;
                reset(stream);
                [actual_ids, num_sc_clusters] = spectral_clustering_with_max_k(Z, K, enable_k);
                if num_sc_clusters == K
                    enable_k = 0;
                end
                [current_ground_lables, ~] = refresh_labels(ground_lables, K);
                [current_cluster_lables, ~] = refresh_labels(actual_ids(1 : n, 1)', K);
                num_current_clusters = length(unique(current_cluster_lables));  
                [acc, nmi, purity, fmeasure, ri, ari] = calculate_dynamic_clustering_results(current_cluster_lables, current_ground_lables, num_current_clusters);
                final_clustering_accs(lambda_idx, dim_idx, beta_idx, wnd_idx) = acc;
                final_clustering_nmis(lambda_idx, dim_idx, beta_idx, wnd_idx) = nmi;
                final_clustering_purities(lambda_idx, dim_idx, beta_idx, wnd_idx) = purity;
                final_clustering_fmeasures(lambda_idx, dim_idx, beta_idx, wnd_idx) = fmeasure;
                final_clustering_ris(lambda_idx, dim_idx, beta_idx, wnd_idx) = ri;
                final_clustering_aris(lambda_idx, dim_idx, beta_idx, wnd_idx) = ari;
                final_clustering_ratios(lambda_idx, dim_idx, beta_idx, wnd_idx) = ratio;
                final_clustering_iters(lambda_idx, dim_idx, beta_idx, wnd_idx) = iter;
                final_clustering_osrc_errors(lambda_idx, dim_idx, beta_idx, wnd_idx, :) = errors;
                                   
                % 3. Online dictionary samples selected 
                [row_size, col_size] = size(X_full);
                Xs_cache = zeros(row_size, col_size);
                num_selected_samples =  0;
                start_index = 1;
                max_iterations = 0;
                for cluster_idx = 1 : num_sc_clusters
                    X_subset = X_full(:, actual_ids == cluster_idx);
                    subset_col_size = size(X_subset, 2);
                    [Z, dic_iter, dic_errors] = online_dic_sample_selection(X_subset, beta);
                    if dic_iter > max_iterations
                        max_iterations = dic_iter;
                        max_iters_errors = dic_errors;
                    end
                    col_positions = zeros(1, length(subset_col_size));                        
                    for col_index = 1 : subset_col_size                            
                        num = length(find(abs(Z(col_index, :)) > 1e-3));
                        if(num > 1)
                            col_positions(1, col_index) = num;
                        end
                    end
                    selected_cols = find(col_positions > 0);
                    if length(selected_cols) > max_num_per_cluster
                       [~, new_col_positions] = sort(col_positions, 'descend');
                       selected_cols = new_col_positions(1:max_num_per_cluster);
                    end                        
                    start_index = num_selected_samples + 1;
                    num_selected_samples = num_selected_samples + length(selected_cols);                        
                    Xs_cache(:, start_index : num_selected_samples) = X_subset(:, selected_cols);                        
                end
                Xs = Xs_cache(:, 1 : num_selected_samples);
                final_clustering_dic_iters(lambda_idx, dim_idx, beta_idx, wnd_idx) = max_iterations;
                final_clustering_dic_errors(lambda_idx, dim_idx, beta_idx, wnd_idx, :) = max_iters_errors;                
                final_clustering_costs(lambda_idx, dim_idx, beta_idx, wnd_idx) = time_cost;
                 disp([wnd_idx, lambda, beta, dim, acc, nmi, purity, fmeasure, ri, ari, ratio, iter, max_iterations]);
                writematrix([wnd_idx, lambda, beta, dim, roundn(acc, -4), roundn(nmi, -4), roundn(fmeasure, -4), ...
                    roundn(ri, -4), roundn(ari, -4), roundn(ratio, -2), roundn(time_cost, -2), iter, max_iterations], final_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
            end
            average_acc =  mean(final_clustering_accs(lambda_idx, dim_idx, beta_idx, :));
            std_acc = std(final_clustering_accs(lambda_idx, dim_idx, beta_idx, :));
            average_nmi =  mean(final_clustering_nmis(lambda_idx, dim_idx, beta_idx, :));
            std_nmi = std(final_clustering_nmis(lambda_idx, dim_idx, beta_idx, :));
            average_purity =  mean(final_clustering_purities(lambda_idx, dim_idx, beta_idx, :));  
            std_purity = std(final_clustering_purities(lambda_idx, dim_idx, beta_idx, :));
            average_fm =  mean(final_clustering_fmeasures(lambda_idx, dim_idx, beta_idx, :));
            std_fm = std(final_clustering_fmeasures(lambda_idx, dim_idx, beta_idx, :));

            average_ri =  mean(final_clustering_ris(lambda_idx, dim_idx, beta_idx, :));
            average_ari =  mean(final_clustering_aris(lambda_idx, dim_idx, beta_idx, :));            
            average_ratio =  mean(final_clustering_ratios(lambda_idx, dim_idx, beta_idx, :));    
            average_iter = mean(final_clustering_iters(lambda_idx, dim_idx, beta_idx, :));    
            average_cost = mean(final_clustering_costs(lambda_idx, dim_idx, beta_idx, :));    
            disp([lambda, beta, dim, average_acc, average_nmi, average_purity, average_fm, average_ri, average_ari, average_ratio, average_iter, average_cost]);
            writematrix([lambda, beta, dim, roundn(average_acc, -4), roundn(std_acc, -4), roundn(average_nmi, -4), roundn(std_nmi, -4), roundn(average_purity, -4), roundn(std_purity, -4), ...
                roundn(average_fm, -4), roundn(std_fm, -4), roundn(average_ri, -4), roundn(average_ari, -4), roundn(average_ratio, -4), roundn(average_iter, -2), roundn(average_cost, -2)], ...
                final_average_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
            writematrix([average_ari, lambda, beta],final_average_ARI, "Delimiter", 'tab', 'WriteMode', 'append'); 
            writematrix([filename, average_ari, lambda, beta],final_averages_ARI, "Delimiter", 'tab', 'WriteMode', 'append'); 
            disp(["AVG ARI",average_ari]);
        end
    end
end
% save(final_average_result_mat, 'final_clustering_accs', 'final_clustering_nmis', 'final_clustering_purities', 'final_clustering_fmeasures', 'final_clustering_ris', 'final_clustering_aris', ...
% 'final_clustering_ratios', 'final_clustering_iters', 'final_clustering_costs', 'final_clustering_osrc_errors', 'final_clustering_dic_iters', 'final_clustering_dic_errors');
