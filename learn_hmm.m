function [bestTR, bestEM] = learn_hmm(train_data, params, varargin)

if nargin == 3
    val_data = varargin{1};
end

% Set up output file
out_file_id = fopen(sprintf('%s_%s_results_%s.txt', ...
                            params.hmm.output_prefix, ...
                            strcat(num2str(params.hmm.latent_dim), '_', ...
                            num2str(params.hmm.num_outputs)), ...
                            datestr(now,'mm-dd-yyyy HH-MM')), 'w');
if nargin == 3
    fprintf(out_file_id, 'restart,elapsed_time,init_trn_loglik_ave,init_trn_loglik_std,init_trn_metric_ave,init_trn_metric_std,init_val_loglik_ave,init_val_loglik_std,init_val_metric_ave,init_val_metric_std,fin_trn_loglik_ave,fin_trn_loglik_std,fin_trn_metric_ave,fin_trn_metric_std,fin_val_loglik_ave,fin_val_loglik_std,fin_val_metric_ave,fin_val_metric_std\n');
else
    fprintf(out_file_id, 'restart,elapsed_time,init_trn_loglik_ave,init_trn_loglik_std,init_trn_metric_ave,init_trn_metric_std,fin_trn_loglik_ave,fin_trn_loglik_std,fin_trn_metric_ave,fin_trn_metric_std\n');
end
bestLL = -Inf;

start_time = tic;

for i = 1:params.hmm.num_restarts
    fprintf('Training HMM Round %i\n\n', i);
    % Initialise params and get performance
    [initTR, initEM] = randomHMM(params.hmm.latent_dim, ...
                                                   params.hmm.num_outputs);
    [init_trn_loglik_ave, init_trn_loglik_std, ...
     init_trn_metric_ave, init_trn_metric_std] = ...
           get_performance_hmm(train_data, initTR, initEM, params.hqmm.train_burn_in);
    if nargin == 3
        [init_val_loglik_ave, init_val_loglik_std, ...
         init_val_metric_ave, init_val_metric_std] = ...
               get_performance_hmm(val_data, initTR, initEM, params.hqmm.val_burn_in);
    end
    
    % Train model and get performance
    [ESTTR, ESTEM] = hmmtrain(train_data(:, ...
                              params.hqmm.train_burn_in+1:end), initTR, ...
                            initEM, 'verbose', true, 'maxiterations', ...
                            params.hmm.max_iterations, 'tolerance',7e-6);
    [fin_trn_loglik_ave, fin_trn_loglik_std, ...
     fin_trn_metric_ave, fin_trn_metric_std] = ...
            get_performance_hmm(train_data, ESTTR, ESTEM, params.hqmm.train_burn_in);
    
    if nargin == 3
        [fin_val_loglik_ave, fin_val_loglik_std, ...
         fin_val_metric_ave, fin_val_metric_std] = ...
                get_performance_hmm(val_data, ESTTR, ESTEM, params.hqmm.val_burn_in);
    end
     
    % Record best seen HMM and save to file
    if nargin < 3
        bestTR = ESTTR;
        bestEM = ESTEM;
        
        fprintf(out_file_id, '%i,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
            i, toc(start_time), ...
            init_trn_loglik_ave, init_trn_loglik_std, ...
            init_trn_metric_ave, init_trn_metric_std, ...
            fin_trn_loglik_ave, fin_trn_loglik_std, ...
            fin_trn_metric_ave, fin_trn_metric_std);
        
    else
        if fin_val_loglik_ave > bestLL
            bestLL = fin_val_loglik_ave;
            bestTR = ESTTR;
            bestEM = ESTEM;
        end
        fprintf(out_file_id, '%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
            i, toc(start_time), ...
            init_trn_loglik_ave, init_trn_loglik_std, ...
            init_trn_metric_ave, init_trn_metric_std, ...
            init_val_loglik_ave, init_val_loglik_std, ...
            init_val_metric_ave, init_val_metric_std, ...
            fin_trn_loglik_ave, fin_trn_loglik_std, ...
            fin_trn_metric_ave, fin_trn_metric_std, ...
            fin_val_loglik_ave, fin_val_loglik_std, ...
            fin_val_metric_ave, fin_val_metric_std);
    end
    
end

save(strcat(params.hmm.output_prefix, '_', ...
       num2str(params.hmm.latent_dim), '_', ...
       num2str(params.hmm.num_outputs), '_TR'), 'bestTR');
save(strcat(params.hmm.output_prefix, '_', ...
       num2str(params.hmm.latent_dim), '_', ...
       num2str(params.hmm.num_outputs), '_EM'), 'bestEM');
end
   