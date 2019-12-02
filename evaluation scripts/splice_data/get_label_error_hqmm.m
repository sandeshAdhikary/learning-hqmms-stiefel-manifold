function [error_ave] = get_label_error_hqmm(data, label, K_ei, K_ie,...
                                                               K_n, params)
% Compute the liklihood for each sequence in data using learned HQMM
% parameters for the EI, IE, and N models. 
% INPUT:
% data: N x D matrix
    % N: number of sequences
    % D: length of a sequence; each element represents a nucleobase
% label: 1 --> EI, 2 --> IE, 3 --> N
% K_ei, K_ie, K_n: Kraus operators for the three models
% params: a struct containing details of training
%
% OUTPUT:
% error_ave: average error across all sequences in data

error = 0.0;
for y = 1:size(data, 1)        
    % Test HQMM
    [ei_loglik, ~, ~, ~] = get_performance_hqmm(data(y, :), ...
                                              K_ei, 'hqmm', 'val', params);
    [ie_loglik, ~, ~, ~] = get_performance_hqmm(data(y, :), ...
                                              K_ie, 'hqmm', 'val', params);
    [n_loglik, ~, ~, ~] = get_performance_hqmm(data(y, :), ...
                                               K_n, 'hqmm', 'val', params);
    
    % Choose model with highest LL as prediction and get error
    [~, pred] = max([ei_loglik, ie_loglik, n_loglik]);
    if label ~= pred
        error = error + 1;
    end

end

error_ave = error/(size(data, 1));

end