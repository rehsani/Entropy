% ****       Entropy Estimation Using Quantile Spacing Approach       ****
% ****           © Hoshin Gupta & Mohammad Reza Ehsani 2021           ****
% ****                                                                ****
% ****        Version 1 written by Mohammad Reza Ehsani in 2020       ****
% ****                                                                ****
% ****                                                                ****
% ****      \                                                 /       ****
% ****       --- THIS IS THE MAIN FUNCTION TO EXECUTE QS ---          ****
% ************************************************************************

function [ h ] = entropy(sample ,alpha ,N_b, N_k)
    n = ceil(alpha * size(sample, 2));
    x_min = min(sample);
    x_max = max(sample);
    sort(sample);
    for ii = 1:N_b
        sample_b = datasample(sample(2:end-1), size(sample, 2));
        for jj = 1:N_k
            X_alpha(jj, :) = datasample(sample_b, n, 'Replace', false);
        end
        X_alpha = sort(X_alpha, 2);
        Z = [x_min mean(X_alpha, 1) x_max];
        dZ = diff(Z);
        H = (1/n) * sum(log(n * dZ));
        h(ii) = H;
    end
end
    
