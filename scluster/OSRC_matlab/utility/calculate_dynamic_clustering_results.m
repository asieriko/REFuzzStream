function [acc, nmi, purity, fmeasure, ri, ari] = calculate_dynamic_clustering_results(currentLabels, groundLables, numClusters)

        groundLables = groundLables(:);
        currentLabels = currentLabels(:);
        
        acc = compute_accuracy(groundLables, currentLabels, numClusters);
        
%         acc2 = accuracy(groundLables, currentLabels);
% 
%         [sortedLabels] = bestMap(groundLables, currentLabels);
%         acc = mean(groundLables==sortedLabels);

        num_classes = length(unique(groundLables));
        class_labels = zeros(1, num_classes);
        for idx =  1 : num_classes
            class_labels(idx) = length(find(groundLables == idx));
        end
        cluster_data = cell(1, numClusters);
        for idx =  1 : numClusters
            cluster_data(1, idx) = { groundLables(currentLabels == idx)' };
        end
        [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);

end
