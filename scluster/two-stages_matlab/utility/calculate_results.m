function [nmi, fmeasure] = calculate_results(num_class_labels, cluster_data)
%class_labels: 1 * N vector
%cluster_data: 1 * N cells
    
    if(size(cluster_data, 2) == 1)
        cluster_data = cluster_data';
    end
    num_class = length(num_class_labels);
    num_cluster = size(cluster_data, 2);    
    n_1 = sum(num_class_labels);
    n = 0;
    for idx = 1 : num_cluster
        data = cluster_data(idx);
        if size(data{1}, 2) == 1
            if size(data{1}, 1) == 1
                n = n + 1;
            else
                error('the size of data should be 1 * N.');
            end
        else
            n = n + size(data{1}, 2);
        end       
    end   
    if n_1 ~= n
        error('error: the numbers of sampes are not coincide.');
    end
    
    % nmi
    ho = 0;
    for idx = 1 : num_class
        rs = num_class_labels(idx) / n;
        ho = ho - rs * log2(rs);
    end
    
    ha = 0;
    hao = 0;
    for i = 1 : num_cluster
        current_cluster = cluster_data(i);
        num_i = size(current_cluster{1}, 2);
        rs = num_i / n;
        if rs > 1e-6
            ha = ha - rs * log2(rs);
        end
 
        hao_tmp = 0;
        for j = 1 : num_class                       
            num_j = length(find(current_cluster{1}(1, :) == j));            
            rs_tmp = num_j / num_i;
            if rs_tmp > 1e-6
                hao_tmp = hao_tmp + rs_tmp * log2(rs_tmp);
            end
        end
        hao = hao - rs * hao_tmp;        
    end
    if (ho + ha) > 0 && (ho - hao) > 0
        nmi = 2 * (ho - hao) / (ho + ha);
    else
        nmi = 0;
    end
%     disp([ho, ha, hao]);
    
    % fmeasure
    fmeasure = 0;
    for i =  1 : num_class       
        fm = 0;
        for j = 1 : num_cluster
            current_cluster = cluster_data(j);            
            num = size(current_cluster{1}, 2);
            len = length(find(current_cluster{1}(1, :) == i));
            precsion = len / num;
            recall = len / num_class_labels(i);
            re = 2 * precsion * recall / (precsion + recall);
            if re > fm
                fm = re;
            end            
        end 
        fmeasure = fmeasure + num_class_labels(i) / n * fm;
    end
    
end

