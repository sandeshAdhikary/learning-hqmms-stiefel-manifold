%% Experiment Script for Synthetic Data Generated by HMM (learning HQMM)
clc
clear classes % Needed when debugging python code

%% Set parameters for models to evaluate
% General Params
params.batch_size = 200;
params.num_epochs = 40;
params.momentum = 0.9;
params.step_size = 0.8;
params.step_size_decay = 0.9;
params.output_prefix = 'evaluation scripts/splice_data/results/hqmm/hqmm';
params.num_folds = 5;

% HQMM
params.hqmm.latent_dim = 4;  % n
params.hqmm.num_outputs = 4; % s
params.hqmm.ancilla_dim = 1; % w
params.hqmm.train_burn_in = 0;
params.hqmm.val_burn_in = 0;
params.hqmm.rho = diag(ones(params.hqmm.latent_dim, 1) / ...
                                                   params.hqmm.latent_dim);

% Dims of Stiefel matrix
num_stiefel_rows = params.hqmm.num_outputs*params.hqmm.ancilla_dim*...
                                                    params.hqmm.latent_dim;
num_stiefel_cols = params.hqmm.latent_dim;
    
%% Set seed for reproducibility
seed = 4;
rng(seed)

%% Load Python Scripts
if count(py.sys.path,'utils/general') == 0
    % If utils/general dir is not in python path, add it
    insert(py.sys.path,int32(0),'utils/general');
end
mod = py.importlib.import_module('qgm_gradient');
py.importlib.reload(mod);

%% Load Data
load('splice_data.mat')

% Shuffle the data
num_ei_examples = size(spliceei, 1);
num_ie_examples = size(spliceie, 1);
num_n_examples = size(splicen, 1);
shuffleEI = spliceei(randsample(1:num_ei_examples, num_ei_examples, ...
                                                                false), :);
shuffleIE = spliceie(randsample(1:num_ie_examples, num_ie_examples, ...
                                                                false), :);
shuffleN = splicen(randsample(1:num_n_examples, num_n_examples, ... 
                                                                false), :);

% Split into folds
ei_splits = crossvalind('Kfold',num_ei_examples,params.num_folds);
ie_splits = crossvalind('Kfold',num_ie_examples,params.num_folds);
n_splits = crossvalind('Kfold',num_n_examples,params.num_folds);

[ei_folds, ie_folds, n_folds] = deal(cell(params.num_folds,1));

for i = 1:params.num_folds
    ei_folds{i} = shuffleEI(ei_splits==i,:);
    ie_folds{i} = shuffleIE(ie_splits==i,:);
    n_folds{i} = shuffleN(n_splits==i,:);
end

% Create list to track error for each fold
[ei_hmm_errors, ie_hmm_errors, n_hmm_errors] = deal( ...
                                                 null(params.num_folds,1));
[ei_hqmm_errors, ie_hqmm_errors, n_hqmm_errors] = deal(...
                                                 null(params.num_folds,1));


%% Train HQMM Models
for k = 1:params.num_folds
       
    % Collect k-1 folds
    val_ei = ei_folds{k};
    val_ie = ie_folds{k};
    val_n = n_folds{k};
    
    train_ei = cell2mat([ei_folds(1:k-1); ei_folds(k+1:end)]);
    train_ie = cell2mat([ie_folds(1:k-1); ie_folds(k+1:end)]);
    train_n = cell2mat([n_folds(1:k-1); n_folds(k+1:end)]);
    
    
    % Train the EI model
    fprintf('[Fold: %d] Training the EI Model...\n',k);

    K_init = random_ortho_mat(num_stiefel_rows, num_stiefel_cols);
    K_init = mat_to_tensor(K_init, params.hqmm.num_outputs, ...
                       params.hqmm.ancilla_dim, params.hqmm.latent_dim,...
                       params.hqmm.latent_dim);
    [K_best_ei, ~] = learn_qgm(train_ei, K_init, params, 'hqmm', val_ei);

    %Train the IE model
    fprintf('\n[Fold: %d] Training the IE Model...\n',k);
    
    K_init = random_ortho_mat(num_stiefel_rows, num_stiefel_cols);
    K_init = mat_to_tensor(K_init, params.hqmm.num_outputs, ...
                       params.hqmm.ancilla_dim, params.hqmm.latent_dim,...
                       params.hqmm.latent_dim);
    [K_best_ie, ~] = learn_qgm(train_ie, K_init, params, 'hqmm', val_ie);
    
    % Train the N model
    fprintf('\n[Fold: %d] Training the N Model...\n',k);
    
    K_init = random_ortho_mat(num_stiefel_rows, num_stiefel_cols);
    K_init = mat_to_tensor(K_init, params.hqmm.num_outputs, ...
                       params.hqmm.ancilla_dim, params.hqmm.latent_dim,...
                       params.hqmm.latent_dim);
    [K_best_n, ~] = learn_qgm(train_n, K_init, params, 'hqmm', val_n);
    
    % Evaluate models on held-out fold
    % Test on EI data
    fprintf('\n[Fold: %d] Evaluating Models on EI Validation Data...\n',k);       
    ei_hqmm_errors(k) = get_label_error_hqmm(val_ei, 1, K_best_ei, ...
                                              K_best_ie, K_best_n, params);
    % Test on IE data 
    fprintf('\n[Fold: %d] Evaluating Models on IE Validation Data...\n',k);      
    ie_hqmm_errors(k) = get_label_error_hqmm(val_ie, 2, K_best_ei, ...
                                              K_best_ie, K_best_n, params);
    % Test on N data    
    fprintf('\n[Fold: %d] Evaluating Models on N Validation Data...\n',k);          
    n_hqmm_errors(k) = get_label_error_hqmm(val_n, 3, K_best_ei, ...
                                              K_best_ie, K_best_n, params);                                
    
    csvwrite(sprintf('%s_%d_%d_%d_ei_errors.csv',params.output_prefix,...
                        params.hqmm.latent_dim, params.hqmm.num_outputs,...
                        params.hqmm.ancilla_dim), ei_hqmm_errors);
    csvwrite(sprintf('%s_%d_%d_%d_ie_errors.csv',params.output_prefix,...
                        params.hqmm.latent_dim, params.hqmm.num_outputs,...
                        params.hqmm.ancilla_dim), ie_hqmm_errors);                                               
    csvwrite(sprintf('%s_%d_%d_%d_n_errors.csv',params.output_prefix,...
                        params.hqmm.latent_dim, params.hqmm.num_outputs,...
                        params.hqmm.ancilla_dim), n_hqmm_errors);                                                  
    
    fprintf('Fold Complete\n')    
end
fprintf('\nHQMM Error Rates:\n');
fprintf('\tEI Error Rate: %f\n',sum(ei_hqmm_errors)/sum(ei_hqmm_errors~=0))
fprintf('\tIE Error Rate: %f\n',sum(ie_hqmm_errors)/sum(ie_hqmm_errors~=0))
fprintf('\tN Error Rate: %f\n', sum(n_hqmm_errors)/sum(n_hqmm_errors~=0))
