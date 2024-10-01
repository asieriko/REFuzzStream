function [result] = calculate_core_similarity(x)

% data: each column represents a sample in data.

x2 = sum(x.^2,2);
distance = bsxfun(@plus,x2, x2')- 2 * x * x';
% nsq= sum(x.^2,2);
% K=bsxfun(@minus,nsq,(2*x)*x');
% K=bsxfun(@plus,nsq.',K);
% re = max(max(abs(K - distance)));
% disp(re);

distance(distance < 1e-5) = 0;

distance = sqrt(distance);


sigma =  mean(distance(:));
k = distance / sigma;
result = exp(-k);

end

