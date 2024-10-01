close all;
clear;
clc;
% r4k: 0.005_0.01_0.05_8: 0.5325
addpath('../data');
addpath('../results');
addpath('utility/');


lambdas =[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];
betas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];

dim = 0;
mus = [0.05];
rhos = [8];

% 1 UG2C5D; 2 Network Intrusion; 3 Keystroke; 4 Forest Cover;
% 5 USPS; 6 COIL-100; 7 Benchmark1_11000; 8 RBF1_40000; 9 DS1
data_index = 1;
switch data_index
    case 1
        lambdas =[0.5];
        betas = [0.01];

        filename = 'uc';
        load('uc_data.mat');
        tssrc_data = uc_data;
        tssrc_labels = uc_labels;
        total_num = size(tssrc_data, 2);
        n = 2000;
        % num_windows = floor(total_num / n);
        num_windows = 50;

    case 2
        lambdas =[1];
        betas = [0.01];

        load('network_data.mat');
        filename = 'network';
        tssrc_data = network_data;
        tssrc_labels = network_labels;
        total_num = size(tssrc_data, 2);
        n = 1000;
        % num_windows = floor(total_num / n);
        num_windows = 50;

    case 3
        lambdas =[0.01];
        betas = [0.5];

        load('key_data.mat');
        filename = 'key';
        tssrc_data = key_data;
        tssrc_labels = key_labels;
        total_num = size(tssrc_data, 2);
        n = 200;
        num_windows = floor(total_num / n);

    case 4
        lambdas =[0.05];
        betas = [0.1];

        load('forest_cover_data.mat');
        filename = 'forest_cover';
        tssrc_data = forest_cover_data;
        tssrc_labels = forest_cover_labels;
        total_num = size(tssrc_data, 2);
        n = 1000;
        % num_windows = floor(total_num / n);
        num_windows = 50;

    case 5
        lambdas =[5e-3];
        betas = [5e-3];

        dim = 50;
        load('usps.mat');
        filename = 'usps';
        tssrc_data = mat2gray(data(:, 2 : end))';
        total_num = size(tssrc_data, 2);
        tssrc_labels = data(1 : total_num, 1)';
        n = 1000;
        num_windows = floor(total_num / n);

    case 6
        lambdas =[1e-3];
        betas = [5e-2];
        dim = 50;
        load('coil100.mat');
        filename = 'coil100';
        tssrc_data = im2double(fea');
        tssrc_labels = gnd';
        total_num = size(tssrc_data, 2);
        rand('state', 100);
        y = randperm(total_num);
        tssrc_data = tssrc_data(:, y);
        tssrc_labels = tssrc_labels(y);
        n = 1000;
        num_windows = floor(total_num / n);

    case 7
        % lambdas =[0.5];
        % betas = [0.01];

        filename = 'b1k';
        load('b1k.mat');
        tssrc_data = inputs;
        tssrc_labels = targets;
        total_num = size(tssrc_data, 2);
        n = 1000;
        % num_windows = floor(total_num / n);
        num_windows = 11;

    case 8
        % lambdas =[0.5];
        % betas = [0.01];

        filename = 'r4k';
        load('r4k.mat');
        tssrc_data = inputs;
        tssrc_labels = targets;
        total_num = size(tssrc_data, 2);
        n = 1000;
        % num_windows = floor(total_num / n);
        num_windows = 40;

    case 9
        % lambdas =[0.5];
        % betas = [0.01];

        filename = 'ds1';
        load('ds1.mat');
        tssrc_data = inputs;
        tssrc_labels = targets;
        total_num = size(tssrc_data, 2);
        n = 100;
        % num_windows = floor(total_num / n);
        num_windows = 8;

end

K = max(tssrc_labels);

lmd_num = length(lambdas);
beta_num = length(betas);
mu_num = length(mus);
rho_num = length(rhos);

combs = lmd_num * beta_num * mu_num * rho_num;
res_mat = zeros(combs,num_windows+4);

final_clustering_accs = zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);
final_clustering_nmis =  zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);
final_clustering_fms =  zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);
final_clustering_sse = zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);
final_clustering_sc = zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);
final_clustering_ratios =  zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);
final_clustering_iters = zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);
final_clustering_costs = zeros(lmd_num, beta_num, mu_num, rho_num, num_windows);

max_iter = 50;
final_z_con_values = zeros(num_windows, max_iter);
final_iterations = zeros(1, num_windows);
final_obj_values = zeros(num_windows, max_iter);

final_result = strcat('../results/', filename, '_final_result.txt');
final_average_result = strcat('../results/', filename, '_final_average_result.txt');
final_result_mat = strcat(filename, '_final_result.mat');
final_average_result_mat = strcat(filename, '_final_average_result.mat');

res_idx = 1;
for lmd_idx = 1 : lmd_num
    lambda = lambdas(lmd_idx);
    for beta_idx = 1 : beta_num
        beta = betas(beta_idx);
        for mu_idx = 1 : mu_num
            mu = mus(mu_idx);
            for rho_idx = 1 : rho_num
                rho = rhos(rho_idx);
                enable_k = 1; % The number of clusters is automatically determined.
                labels_mat = zeros(n, num_windows * 2);
                aris = zeros(num_windows,1);
                ris = zeros(num_windows,1);
                var_names = strcat(num2str(lambdas(lmd_idx)),'_', num2str(betas(beta_idx)),'_', num2str(mus(mu_idx)),'_', num2str(rhos(rho_idx)));
                % disp(var_names);
                for wnd_idx = 1 : num_windows
                    start_idx = (wnd_idx - 1) * n + 1;
                    X = tssrc_data(:, start_idx : start_idx + n - 1);
                    for i = 1 : n
                        X(:, i) = X(:, i) ./ max(1e-12,norm(X(:, i)));

                    end
                    ground_lables = tssrc_labels(start_idx : start_idx + n - 1);
                    if wnd_idx == 1
                        X_full = X;
                    else
                        X_full = [X, Xs];
                    end
                    if dim > 1e-6
                        [eigen_vector, ~, ~] = f_pca(X_full, dim);
                        XX = normc(eigen_vector' * X_full);
                    else
                        XX = normc(X_full);
                    end
                    tic;
                    %                     [Zc, D, iter] = trsc(normc(X_full), lambda, beta, mu, rho);
                    [Zc, D, iter, obj_values, z_con_values] = tssrc(XX, lambda, beta, mu, rho);
                    time_cost1 = toc;
                    final_z_con_values(wnd_idx, :) = z_con_values;
                    final_iterations(wnd_idx) = iter;
                    final_obj_values(wnd_idx, :)  = obj_values;
                    Z = abs(Zc) + abs(Zc');
                    total_num = size(X_full, 2);
                    ratio = length(find(abs(Z) > 1e-6)) / (total_num * total_num);
                    stream = RandStream.getGlobalStream;
                    reset(stream);
                    [actual_ids, num_sc_clusters, sse, sc] = spectral_clustering_with_max_k(Z, K, enable_k, wnd_idx, n);
                    % disp([num_current_clusters, num_sc_clusters]);
                    if num_sc_clusters == K
                        enable_k = 0;
                    end
                    if wnd_idx == 1
                        % the representative objects used in the next window
                        Xs = X;
                    else
                        % the representative objects used in the next window
                        Xs = construct_representative_objects(X_full, D, Z, n, actual_ids, num_sc_clusters);
                    end

                    [current_ground_lables, num_current_clusters] = refresh_labels(ground_lables, K);
                    [actual_ids, ~] = refresh_labels(actual_ids(1 : n), K);

                    % 1. acc
                    acc = compute_accuracy(current_ground_lables, actual_ids, num_sc_clusters);

                    cluster_data = cell(1, num_sc_clusters);
                    class_labels = zeros(1, num_current_clusters);
                    for pos_idx =  1 : num_sc_clusters
                        cluster_data(1, pos_idx) = { current_ground_lables(actual_ids == pos_idx) };
                    end
                    for idx =  1 : num_current_clusters  % the last cluster is -1, for outliers
                        class_labels(idx) = length(find(current_ground_lables == idx));
                    end
                    labels_mat(:,(2*wnd_idx)-1) = current_ground_lables;
                    labels_mat(:,(2*wnd_idx)) = actual_ids'; 
                    [RI, ARI] = randindex(current_ground_lables, actual_ids);
                    aris(wnd_idx) = ARI;
                    ris(wnd_idx) = ARI;
                    % disp([num2str(wnd_idx * n) ,' RI: ', num2str(RI), ' ARI: ', num2str(ARI)]);
                    % 2. nmi and fmeasure
                    % [nmi, fmeasure] = calculate_results(class_labels, cluster_data);
                    % time_cost = toc;
                    % final_clustering_accs(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = acc;
                    % final_clustering_nmis(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = nmi;
                    % final_clustering_fms(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = fmeasure;
                    % final_clustering_sse(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = sse;
                    % final_clustering_sc(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = sc;
                    % final_clustering_ratios(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = ratio;
                    % final_clustering_iters(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = iter;
                    % final_clustering_costs(lmd_idx, beta_idx, mu_idx, rho_idx, wnd_idx) = time_cost;
                    % disp([wnd_idx, num_current_clusters, num_sc_clusters, acc, nmi, fmeasure, sse, sc, ratio, iter, time_cost]);
                    % writematrix([wnd_idx, num_current_clusters, num_sc_clusters, roundn(acc, -4), roundn(nmi, -4), roundn(fmeasure, -4), ...
                    %     roundn(sse, -4), roundn(sc, -4), roundn(ratio, -2), roundn(time_cost, -2), iter], final_result, "Delimiter", 'tab', 'WriteMode', 'append');
                end
                res_mat(res_idx,1) = lambdas(lmd_idx);
                res_mat(res_idx,2) = betas(beta_idx);
                res_mat(res_idx,3) = mus(mu_idx);
                res_mat(res_idx,4) = rhos(rho_idx);
                res_mat(res_idx,5:num_windows+4) = aris;
                disp(strcat(filename, ": ",var_names, ": ",num2str(mean(aris))));
                writematrix(labels_mat, strcat('two-stages/results/', filename,'_', var_names, '_labels.csv'));
                writematrix(aris, strcat('two-stages/results/', filename,'_', var_names, '_ARI.csv'));
                writematrix(ris, strcat('two-stages/results/', filename,'_', var_names, '_RI.csv'));
                res_idx = res_idx + 1;
                %                 save(final_result_mat, 'final_iterations', 'final_z_con_values', 'final_obj_values');
                % average_acc =  mean(final_clustering_accs(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % average_nmi =  mean(final_clustering_nmis(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % average_fm =  mean(final_clustering_fms(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % average_sc =  mean(final_clustering_sc(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % average_sse =  mean(final_clustering_sse(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % average_ratio =  mean(final_clustering_ratios(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % average_iter = mean(final_clustering_iters(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % average_cost = mean(final_clustering_costs(lmd_idx, beta_idx, mu_idx, rho_idx, :));
                % disp([lambda, beta, mu, rho, average_acc, average_nmi, average_fm, average_sc, average_sse, average_ratio, average_iter, average_cost]);
                % writematrix([lambda, beta, mu, rho, roundn(average_acc, -4), roundn(average_nmi, -4), roundn(average_fm, -4), roundn(average_sc, -4), ...
                %     roundn(average_sse, -4), roundn(average_ratio, -4), roundn(average_iter, -2), roundn(average_cost, -2)], final_average_result, "Delimiter", 'tab', 'WriteMode', 'append');
            end
        end
    end
end
writematrix(res_mat, strcat('two-stages/results/', filename, '_RI.csv'));
%save(final_average_result_mat, 'final_clustering_accs', 'final_clustering_nmis', 'final_clustering_fms', 'final_clustering_sc', 'final_clustering_sse', 'final_clustering_ratios', 'final_clustering_iters');

