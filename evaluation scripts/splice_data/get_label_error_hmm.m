function [error_ave] = get_label_error_hmm(data, label, trans_ei, ...
                                           emit_ei, trans_ie, emit_ie, ...
                                           trans_n, emit_n)
% Compute the likelihood for each sequence in data using learned HMM
% parameters for the EI, IE, and N models. 
% INPUT:
% data: N x D matrix
    % N: number of sequences
    % D: length of a sequence; each element represents a nucleobase
% label: 1 --> EI, 2 --> IE, 3 --> N
% trans_*, emit_*: row-stochastic transition and emission matrices
% params: a struct containing details of training
%
% OUTPUT:
% error_ave: average error across all sequences in data

error = 0.0;
for y = 1:size(data, 1)        
    % Test HMM
    [ei_loglik, ~, ~, ~] = get_performance_hmm(data(y, :), trans_ei, emit_ei, 0);
    [ie_loglik, ~, ~, ~] = get_performance_hmm(data(y, :), trans_ie, emit_ie, 0);
    [n_loglik, ~, ~, ~] = get_performance_hmm(data(y, :), trans_n, emit_n, 0);
    
    % Choose model with highest LL as prediction and get error
    [~, pred] = max([ei_loglik, ie_loglik, n_loglik]);
    if label ~= pred
        error = error + 1;
    end

end

error_ave = error/(size(data, 1));

end