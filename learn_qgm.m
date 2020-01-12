function [K_best, hist] = learn_qgm(train_data, K_init, params, ...
                                                      model_type, varargin)
% INPUTS:
% train_data: Matrix with the following dimensions
    % HQMM: num_data x seq_len
    % QNB: num_data x (num_features + 1). The first col should have labels
% K_init = 4 mode tensor with the following dimensions:
    % HQMM: Input tensor dim = (Num Obs) x (W) x (Latent State Dim) x 
    %                                                    (Latent State Dim)
    % QNB: Input tensor dim = (Num Obs) x (Num Features) x (Num Labels) x 
    %                                                          (Num_labels)
% params: struct containing model parameters
% model_type = String specifying which QGM
    % HQMM: 'hqmm'
    % QNB: 'qnb'      
% Optional Arguments:
    % val_data: validation data
%
% OUTPUTS:
% K_best: Tensor (same shape as K_init) of learned Kraus operators. If a
%         validation set is provided, return the best-performing Kraus
%         operators on the validation set, else return Kraus operators 
%         at the end of training.
% hist: struct with info on training
   
    %% Initialize Parameters
    
    % Set model parameters
    if strcmp(model_type, 'hqmm')
        % Set dims of tensor of Kraus operators
        dim1 = params.hqmm.num_outputs;
        dim2 = params.hqmm.ancilla_dim;
        dim3 = params.hqmm.latent_dim;
        dim4 = dim3;
        stiefel_nrows = dim1 * dim2 * dim3;
        stiefel_ncols = dim3;
        
        num_train = size(train_data, 1);
        rho = params.hqmm.rho;

    elseif strcmp(model_type, 'qnb')
        error('QNB not coded.');
    else 
        error('Unknown Model Type');
    end
    
    % Get validation data (if provided)
    if (nargin > 4)
        val_data = varargin{1};
    end

    % Set hyperparameters
    num_batches = ceil(num_train/params.batch_size);
    step_size = params.step_size;
    momentum = params.momentum;

    % Set up output file
    out_file_id = fopen(sprintf('%s_%s_results_%s.txt', ...
              params.output_prefix, strcat(num2str(dim3), '_', ...
                                    num2str(dim1), '_', num2str(dim2)), ...
                                    datestr(now,'mm-dd-yyyy HH-MM')), 'w');
   
    % Start timer                        
    start_time = tic;                      
 
    % Track best parameters & likelihood
    K_learned = K_init;
    K_best = K_learned;
    best_val_LL = -Inf;
    
    % Create zero-vectors to hold metrics
    [hist.train_LL_ave, hist.train_LL_std, ...
     hist.val_LL_ave, hist.val_LL_std, ...
     hist.train_metric_ave, hist.train_metric_std, ...
     hist.val_metric_ave, hist.val_metric_std, ...
     hist.G_diff, hist.K_diff, hist.G_norm, hist.ortho_error] = ...
                                     deal(zeros(1, params.num_epochs + 1));
    
    % Record likelihood and metrics prior to training
    [hist.train_LL_ave(1), hist.train_LL_std(1), ...
     hist.train_metric_ave(1), hist.train_metric_std(1)] = ...
                            get_performance_hqmm(train_data, K_learned, ...
                                              model_type, 'train', params);
    fprintf('Initial training Neg Log Likelihood: %f, Metric: %f\n', ...
                          -hist.train_LL_ave(1), hist.train_metric_ave(1));
                      
    if (nargin > 4)
        % if validation data provided
        [hist.val_LL_ave(1), hist.val_LL_std(1), ...
         hist.val_metric_ave(1), hist.val_metric_std(1)] = ...
                              get_performance_hqmm(val_data, K_learned, ...
                                                model_type, 'val', params);
        fprintf('Initial Val Neg Log Likelihood: %f, Metric: %f\n', ...
                              -hist.val_LL_ave(1), hist.val_metric_ave(1));
    end
    
    % Record initial orthogonality error
    K_learned_mat = tensor_to_mat(K_learned, stiefel_nrows, stiefel_ncols);
    init_ortho_error = norm(eye(stiefel_ncols) - ...
                                  ctranspose(K_learned_mat)*K_learned_mat);
    
    % Write initial values to the output file
    if (nargin > 4)
        % If validation data provided
        fprintf(out_file_id,'epochs,time,step_size,train_neg_LL,val_neg_LL,train_metric_ave,val_metric_ave,ortho_error_ave,G_norm,G_diff,K_diff,grad_time,stiefel_time\n');
        fprintf(out_file_id,...
                '%d,%.2f,%.2f,%f,%f,%f,%f,%.2e,%f,%.3f,%.3f,%.3f,%.2e\n', ...
                0, toc(start_time), step_size, ...
                -hist.train_LL_ave(1), -hist.val_LL_ave(1), ...
                hist.train_metric_ave(1), hist.val_metric_ave(1), ...
                init_ortho_error, 0, 0, 0, 0, 0);  
    else
        % If validation data not provided
        fprintf(out_file_id,'epochs,time,step_size,train_neg_LL,train_metric_ave,ortho_error_ave,G_norm,G_diff,K_diff,grad_time,stiefel_time\n');
        fprintf(out_file_id,'%d,%.2f,%.2f,%f,%f,%.2e,%f,%.3f,%.3f,%.3f,%.2e\n',...
            0, toc(start_time), step_size, ...
            -hist.train_LL_ave(1), ...
            hist.train_metric_ave(1), ...
            init_ortho_error, 0, 0, 0, 0, 0);    
    end

    % Initialize metrics; for recording only
    G_epoch_old = 0;
    K_epoch_old = K_learned_mat;

    %% Begin training
    for epoch = 1:params.num_epochs
       
        fprintf('Epoch: %d\n', epoch);
        fprintf('Step Size: %f\n', step_size);

        % Epoch stats
        G_batch_old = 0; % For momentum
        running_G_norm = 0;
        running_grad_time = 0; 
        running_stiefel_time = 0;
        ortho_error_running_total = 0;
        G_running_total = zeros(stiefel_nrows, stiefel_ncols);
        
        % Shuffle training data
        shuffled_indices = randperm(num_train);
        
        for b = 1:num_batches
            % Prepare batch
            fprintf('\tProcessing batch %i of %i\n', b, num_batches);
            start_index = (b-1)*params.batch_size+1;
            
            if size(shuffled_indices(start_index : end), 2) < ...
                                                          params.batch_size
                batch = train_data(shuffled_indices(start_index : end)',:);
            else
                selected_indices = start_index : b*params.batch_size;
                batch = train_data(shuffled_indices(selected_indices)', :);
            end
            
            % Compute Gradient
            fprintf('\tComputing Gradient...\n')
            grad_start_time = tic;
            
            K_learned_real = real(K_learned);
            K_learned_imag = imag(K_learned);
            
            if strcmp(model_type, 'hqmm')
                rho_real = real(rho);
                rho_imag = imag(rho);
                G_tuple = py.qgm_gradient.get_hqmm_gradient( ...
                                        K_learned_real, K_learned_imag, ...
                                        rho_real, rho_imag, batch, ...
                                        params.hqmm.train_burn_in, 'loss');
            elseif strcmp(model_type, 'qnb')
                % TODO: Add Gradient Script for QNB
                error('QNB Not Coded');
            end
            
            grad_stop_time = toc(grad_start_time);
            fprintf('\tComputed Gradient in %f seconds\n', grad_stop_time)
            G_tensor = double(nparray_to_matlab(G_tuple{1})) +...
                1i*double(nparray_to_matlab(G_tuple{2}));      
                       
            % Reshape G_tensor and K_learned to Stiefel matrices
            G_mat = tensor_to_mat(G_tensor, stiefel_nrows, stiefel_ncols);
            K_learned_mat = tensor_to_mat(K_learned, stiefel_nrows, ...
                                                            stiefel_ncols);
         
            % Normalize the gradient
            G_norm = norm(G_mat, 2);
            running_G_norm = running_G_norm + G_norm;
            G_mat = G_mat / G_norm;
            
            % Apply momentum
            G_mat = (momentum)*G_batch_old + (1-momentum)*G_mat;
            G_mat = G_mat/norm(G_mat, 2); % re-normalize the gradient
            G_batch_old = G_mat;
            
                       
            % Supply Stiefel matrix to Update
            fprintf('\t Performing Stiefel Update\n')
            stiefel_start_time = tic;
            K_learned_mat =  stiefel_update(K_learned_mat, G_mat, ...
                                            step_size);
            stiefel_stop_time = toc(stiefel_start_time);
            fprintf('\t Completed Stiefel Update in %f seconds\n',...
                    stiefel_stop_time);
            
            % Check orthogonality error of updated K_learned_mat
            ortho_error = norm(ctranspose(K_learned_mat)*...
                                     K_learned_mat - eye(stiefel_ncols),'fro');
            fprintf('\t Orthogonality Error: %e\n', ortho_error)

            % Convert K_learned_mat back to a tensor
            K_learned = mat_to_tensor(K_learned_mat, dim1, dim2, dim3, ...
                                                                     dim4);
            
            % Update running totals for epoch
            running_grad_time = running_grad_time + grad_stop_time;            
            running_stiefel_time = running_stiefel_time + ...
                                                         stiefel_stop_time;
            ortho_error_running_total = ortho_error_running_total + ...
                                        ortho_error;
            G_running_total = G_running_total + G_mat;

        end
        
        % Compute averages for epoch
        G_norm_ave = running_G_norm/num_batches;
        grad_time_ave = running_grad_time/num_batches;
        stiefel_time_ave = running_stiefel_time/num_batches;
        ortho_error_ave = ortho_error_running_total/num_batches;
        G_epoch_ave = G_running_total/num_batches;
        
        % Record Training Performance
        [hist.train_LL_ave(epoch+1), ...
         hist.train_LL_std(epoch+1), ...
         hist.train_metric_ave(epoch+1),...
         hist.train_metric_std(epoch+1)] = get_performance_hqmm(train_data,...
                                   K_learned, model_type, 'train', params);
         fprintf('Training Neg Log Likelihood: %f, Metric: %f\n', ...
                                           -hist.train_LL_ave(epoch+1), ...
                                           hist.train_metric_ave(epoch+1));
  
        if nargin > 4
            % Record validation performance (when validation data provided)
            [hist.val_LL_ave(epoch+1), ...
             hist.val_LL_std(epoch+1), ...
             hist.val_metric_ave(epoch+1), ...
             hist.val_metric_std(epoch+1)] = get_performance_hqmm(val_data, ...
                                     K_learned, model_type, 'val', params);
             fprintf('Val Neg Log Likelihood: %f, Metric: %f\n', ...
                                            -hist.val_LL_ave(epoch+1), ...
                                             hist.val_metric_ave(epoch+1));
                                          
             % Save model if it is the best thus far
             if hist.val_LL_ave(epoch+1) > best_val_LL
                best_val_LL = hist.val_LL_ave(epoch+1);
                K_best = K_learned;
                save(strcat(params.output_prefix, '_', num2str(dim3), ...
                            '_', num2str(dim1), '_', num2str(dim2), ...
                            '_K_learned.mat'), 'K_best'); 
             end
        else
            % If validation data not provided
            % Save K_learned at the end of every epoch
            K_best = K_learned;
            save(strcat(params.output_prefix,  '_', num2str(dim3), ...
                        '_', num2str(dim1), '_', num2str(dim2), ...
                        '_K_learned.mat'), 'K_best'); 
        end

        % Record other metrics
        hist.G_norm(1, epoch+1) = G_norm_ave;
        hist.K_diff(1, epoch+1) = norm(K_learned_mat - K_epoch_old, 'fro');
        K_epoch_old = K_learned_mat;
        hist.G_diff(1, epoch+1) = norm(G_epoch_ave - G_epoch_old, 'fro');
        G_epoch_old = G_epoch_ave;
        hist.ortho_error(1,epoch+1) = ortho_error_ave;
              
        % Display Epoch time
        toc(start_time)
        
        % Write results to output file
        if nargin > 4 
            % if validation data provided
            fprintf(out_file_id, '%d,%.2f,%.2f,%f,%f,%f,%f,%.2e,%f,%.3f,%.3f,%.3f,%.2e\n', ...
                    epoch, toc(start_time), step_size, ...
                    -hist.train_LL_ave(epoch+1), ...
                    -hist.val_LL_ave(epoch+1), ...
                    hist.train_metric_ave(epoch+1), ...
                    hist.val_metric_ave(epoch+1), ortho_error_ave, ...
                    hist.G_norm(1, epoch+1), ...
                    hist.G_diff(1, epoch+1), hist.K_diff(1, epoch+1), ...
                    grad_time_ave, stiefel_time_ave);    
        else
            % if validation data not provided
            fprintf(out_file_id, '%d,%.2f,%.2f,%f,%f,%.2e,%f,%.3f,%.3f,%.3f,%.2e\n',...
                    epoch, toc(start_time), step_size, ...
                    -hist.train_LL_ave(epoch+1), ...
                    hist.train_metric_ave(epoch+1), ...
                    ortho_error_ave, ...
                    hist.G_norm(1, epoch+1), ...
                    hist.G_diff(1, epoch+1), hist.K_diff(1, epoch+1), ...
                    grad_time_ave, stiefel_time_ave);   
        end
        
        % Update learning rate
        step_size = step_size*params.step_size_decay;        
    
    end
    
    % Close the output file
    fclose(out_file_id);
end
