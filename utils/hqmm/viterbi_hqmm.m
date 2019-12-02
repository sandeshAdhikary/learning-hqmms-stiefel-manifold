function seq = viterbi_hqmm(K_cell, obs, num_hidden, START_ID)

trajectories = zeros(num_hidden, size(obs, 2));
max_probs = zeros(num_hidden, size(obs, 2));

init_probs = zeros(num_hidden, 1);
for w = 1:size(K_cell, 2)
    init_K = K_cell{obs(1), w};
    init_probs = init_probs + (init_K(:, START_ID) .* conj(init_K(:, START_ID)));
end
max_probs(:, 1) = log(init_probs);
trajectories(:, 1) = ones(num_hidden, 1) * START_ID;

for i = 2:size(obs, 2)
    probs = zeros(num_hidden, 1);
    for w = 1:size(K_cell, 2)
        next_K = K_cell{obs(i), w};
        probs = probs + (next_K .* conj(next_K));
    end
    [max_probs(:, i), trajectories(:, i)] = max(log(probs) + max_probs(:, i-1)', [], 2);
end

[~, final_ind] = max(max_probs(:, end));

seq = zeros(size(obs));
seq(size(obs, 2)) = final_ind;
prev_ind = final_ind;

for i = size(obs, 2):-1:2
    seq(i-1) = trajectories(prev_ind, i);
    prev_ind = seq(i-1);
end

end