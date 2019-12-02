%% Experiment Script for Synthetic Data Generated by HMM
clc
clear classes % Needed when debugging python code

%% Set seed for reproducibility
seed = 15;
rng(seed)

%% Set parameters for models to evaluate
% General Params
params.batch_size = 20;
params.num_epochs = 60;
params.momentum = 0.9;
params.step_size = 0.75;
params.step_size_decay = 0.92;
params.output_prefix = 'evaluation scripts/synthetic_hmm/results/hqmm';

% HQMM
params.hqmm.latent_dim = 6;  % n
params.hqmm.num_outputs = 6; % s
params.hqmm.ancilla_dim = 6; % w
params.hqmm.train_burn_in = 100;
params.hqmm.val_burn_in = 1000;
params.hqmm.rho = RandomDensityMatrix(params.hqmm.latent_dim);

% HMM (if HMM needs to be trained for reference)
params.hmm.latent_dim = 6;
params.hmm.num_outputs = 6;
params.hmm.num_restarts = 5;
params.hmm.max_iterations = 600;
params.hmm.output_prefix = 'evaluation scripts/synthetic_hmm/results/hmm';

%% Load Python Scripts
if count(py.sys.path,'utils/general') == 0
    % If utils/general dir is not in python path, add it
    insert(py.sys.path,int32(0),'utils/general');
end
mod = py.importlib.import_module('qgm_gradient');
py.importlib.reload(mod);

%% Load Data
% Load Synthetic HQMM generated data
load('synthetic_hmm_data.mat'); 
train_data = reshape(data.trn_data', 300, 200)';
val_data = data.val_data;

%% Generate a random orthonormal matrix as initial K
num_stiefel_rows = params.hqmm.num_outputs*params.hqmm.ancilla_dim*...
                                                    params.hqmm.latent_dim;
num_stiefel_cols = params.hqmm.latent_dim;
K_init = random_ortho_mat(num_stiefel_rows,num_stiefel_cols);
K_init = mat_to_tensor(K_init, params.hqmm.num_outputs, ...
                       params.hqmm.ancilla_dim, params.hqmm.latent_dim,...
                       params.hqmm.latent_dim);
                   
%% Learn HQMM and HMM (for reference)
[K_best, hist] = learn_qgm(train_data, K_init, params, 'hqmm', val_data);
[bestTR, bestEM] = learn_hmm(train_data, params, val_data);

% Evaluate learned HQMM model
[hqmm_trn_loglik, hqmm_trn_loglik_std, ...
 hqmm_trn_metric, hqmm_trn_metric_std] = ...
              get_performance_hqmm(train_data, K_best, 'hqmm', 'train', params);
[hqmm_val_loglik, hqmm_val_loglik_std, ...
 hqmm_val_metric, hqmm_val_metric_std] = ...
                  get_performance_hqmm(val_data, K_best, 'hqmm', 'val', params);
test_data = data.test_data;
[hqmm_te_loglik, hqmm_te_loglik_std, ...
 hqmm_te_metric, hqmm_te_metric_std] = ...
                 get_performance_hqmm(test_data, K_best, 'hqmm', 'val', params);

% Evaluate learned HMM model
[hmm_trn_loglik, hmm_trn_loglik_std, ...
hmm_trn_metric, hmm_trn_metric_std] = ...
          get_performance_hmm(train_data, bestTR, bestEM, params.hqmm.train_burn_in);
[hmm_val_loglik, hmm_val_loglik_std, ...
hmm_val_metric, hmm_val_metric_std] = ...
              get_performance_hmm(val_data, bestTR, bestEM, params.hqmm.val_burn_in);
test_data = data.test_data;
[hmm_te_loglik, hmm_te_loglik_std, ...
hmm_te_metric, hmm_te_metric_std] = ...
              get_performance_hmm(test_data, bestTR, bestEM, params.hqmm.val_burn_in);   


% Write to file
results_filename = sprintf('%s_synth_results_%d-%d-%d.csv',...
                    params.hmm.output_prefix, params.hqmm.latent_dim, ...
                    params.hqmm.num_outputs, params.hqmm.ancilla_dim);
                                           
results_fileID = fopen(results_filename, 'w');
fprintf(results_fileID, 'model,n,s,w,train_metric_ave,train_metric_std,val_metric_ave,val_metric_std,test_metric_ave,test_metric_std\n');
fprintf(results_fileID, '%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
        'hqmm', params.hqmm.latent_dim, ...
        params.hqmm.num_outputs, params.hqmm.ancilla_dim, ...
        hqmm_trn_metric, hqmm_trn_metric_std, ...
        hqmm_val_metric, hqmm_val_metric_std, ...
        hqmm_te_metric, hqmm_te_metric_std);
fprintf(results_fileID, '%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f', ...
        'hmm', params.hqmm.latent_dim, ...
        params.hqmm.num_outputs, 0, ...
        hmm_trn_metric, hmm_trn_metric_std, ...
        hmm_val_metric, hmm_val_metric_std, ...
        hmm_te_metric, hmm_te_metric_std);