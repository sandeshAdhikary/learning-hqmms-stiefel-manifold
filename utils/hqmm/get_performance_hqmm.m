function [loglik, loglik_std, metric, metric_std] = ...
                  get_performance_hqmm(data, K_tensor, model_type, mode, params)
% Compute log-liklihood (LL) and accuracy metric for the given data
% INPUT:
% data: N x L matrix; N data points, L is dim of sequence
%       Note: If data has feats and labels, first col of data is 
%             labels and remaining cols are feats
% K_tensor: Model parameters (Kraus operators) as 4-tensor
% model_type: hqmm - hidden quantum Markov model
% mode: to use different burn in for training/val data
%       'train': data is training data
%       'val': data is val data
% params: a struct containing details of training
%
% OUTPUT: 
% loglik: mean log likelihood of the data under the model
% loglik_std: std dev of log likelihood of data under the model
% metric: average performance according to some metric
%         hqmm: Descriptive Accuracy (DA) score
% metric_std: std dev performance according to some metric

% Get loglikelihood info
loglik_list = zeros(size(data, 1), 1);
for i = 1:size(data,1)
    if strcmp(model_type, 'hqmm')
        % Convert Tensor to Cell to use with loglik function
        nrows = params.hqmm.num_outputs * params.hqmm.ancilla_dim * ...
                                                    params.hqmm.latent_dim;
        ncols = params.hqmm.latent_dim;
        K_cell = transpose(reshape(mat2cell( ...
                                 tensor_to_mat(K_tensor, nrows, ncols), ...
                          ones(nrows/ncols, 1)*params.hqmm.latent_dim), ...
                        params.hqmm.ancilla_dim, params.hqmm.num_outputs));
        % Compute log likelihood for given sequence
        if strcmp(mode, 'train')
            loglik_list(i) = loglik_hqmm(data(i,:)', K_cell, ...
                               params.hqmm.rho, params.hqmm.train_burn_in);
        elseif strcmp(mode, 'val')
            loglik_list(i) = loglik_hqmm(data(i,:)', K_cell, ...
                                 params.hqmm.rho, params.hqmm.val_burn_in);
        else
            error('Unknown mode!')
        end
    elseif strcmp(model_type, 'qnb')
        error('QNB performance calculation not coded yet')
    else
        error('Unknown Model Type!')
    end
end
    
loglik = mean(loglik_list);
loglik_std = std(loglik_list);

% Get metric
if strcmp(model_type, 'hqmm')
    if strcmp(mode, 'train')
        [metric, metric_std] = getDA(loglik_list, ...
                                params.hqmm.num_outputs, ...
                                size(data, 2) - params.hqmm.train_burn_in);
    elseif strcmp(mode, 'val')
        [metric, metric_std] = getDA(loglik_list, ...
                                params.hqmm.num_outputs, ...
                                size(data, 2) - params.hqmm.val_burn_in);
    else
        error('Unknown Mode!')
    end
elseif strcmp(model_type, 'hqmm')
    error('Nah')
else
    error('Unknown Model Type!')
end

end
