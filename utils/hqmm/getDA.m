function [da, da_std] = getDA(loglik_list, output_dim, seq_length)
%% Function to compute the description accuracy
% INPUT:
%   LL: vector of log likelihoods of each sequence
%   output_dim: dimension of output space
%   seq_length: length of sequence

da_list = 1 + loglik_list/(log(output_dim) * seq_length);
 
da = mean(da_list);
da_std = std(da_list);

end