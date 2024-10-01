function [RI, ARI] = randindex(labels1, labels2)
    % https://www.mathworks.com/matlabcentral/fileexchange/130779-rand-and-adjusted-rand-index-calculator-for-cluster-analysis
    % The function calculates the Rand index (RI) and Adjusted Rand index (ARI)
    % between two label assignments: labels1 and labels2.
    
    N = numel(labels1);  % Get the number of elements in the label vector
    
    % Initialize the four quantities: TP (true positive), FN (false negative), FP (false positive), TN (true negative)
    TP = 0; FN = 0; FP = 0; TN = 0;
    
    % Calculate TP, FN, FP and TN
    for i = 1:N-1
        for j = i+1:N
            if (labels1(i) == labels1(j)) && (labels2(i) == labels2(j))  % TP: Both labels1 and labels2 have the same class for samples i and j
                TP = TP + 1;
            elseif (labels1(i) == labels1(j)) && (labels2(i) ~= labels2(j))  % FN: labels1 have the same class for samples i and j, but labels2 not
                FN = FN + 1;
            elseif (labels1(i) ~= labels1(j)) && (labels2(i) == labels2(j))  % FP: labels2 have the same class for samples i and j, but labels1 not
                FP = FP + 1;
            else   % TN: Both labels1 and labels2 have different classes for samples i and j
                TN = TN + 1;
            end
        end
    end
    
    % Calculate Rand Index (RI)
    RI = (TP + TN) / (TP + FP + FN + TN);
    
    % Calculate Adjusted Rand Index (ARI)
    try
        C = confusionmat(labels1, labels2);  % Use built-in function to generate confusion matrix
    catch
        C = myConfusionMat(labels1, labels2);  % If built-in function is not available, use the self-defined function
    end
    
    % Compute necessary quantities for ARI calculation
    sum_C = sum(C(:));
    sum_C2 = sum_C * (sum_C - 1);
    sum_rows = sum(C, 2);
    sum_rows2 = sum(sum_rows .* (sum_rows - 1));
    sum_cols = sum(C, 1);
    sum_cols2 = sum(sum_cols .* (sum_cols - 1));
    sum_Cij2 = sum(sum(C .* (C - 1)));
    
    % Compute ARI
    ARI = 2 * (sum_Cij2 - sum_rows2 * sum_cols2 / sum_C2) / ...
        ((sum_rows2 + sum_cols2) - 2 * sum_rows2 * sum_cols2 / sum_C2);
end
function C = myConfusionMat(g1, g2)
    % This function generates a confusion matrix if the built-in function confusionmat is not available.
    % It takes two input vectors, g1 and g2, which represent two different label assignments.
    
    groups = unique([g1;g2]);  % Get the unique groups in g1 and g2
    
    C = zeros(length(groups));  % Initialize the confusion matrix with zeros
    
    % Calculate the confusion matrix
    for i = 1:length(groups)
        for j = 1:length(groups)
            C(i,j) = sum(g1 == groups(i) & g2 == groups(j));  % Count the number of samples that belong to group i in g1 and group j in g2
        end
    end
end