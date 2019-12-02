function [loglik_ave, loglik_std, da_ave, da_std] = get_performance_hmm(data, ...
                                                           TR, EM, burn_in)

loglik_list = zeros(size(data, 1), 1);
for i = 1:size(data, 1)
    [~, log_tot_prob] = hmmdecode(data(i, :), TR, EM);
    loglik_list(i) = log_tot_prob;
    if burn_in ~= 0
        [~, log_burn_in] = hmmdecode(data(i, 1:burn_in), TR, EM);
        loglik_list(i) = loglik_list(i) - log_burn_in;
    end
end

loglik_ave = mean(loglik_list);
loglik_std = std(loglik_list);
[da_ave, da_std] = getDA(loglik_list, size(EM, 2), size(data, 2)- burn_in);
                          
end