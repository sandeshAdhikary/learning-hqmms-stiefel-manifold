function K = HMMtoHQMM(A, C)

dim = size(C, 2);
output_dim = size(C,1);

K = cell(output_dim, dim);

for o = 1:output_dim
    sq_obs_op = sqrt(diag(C(o,:)) * A);
    for n = 1:dim
        temp = zeros(dim, dim);
        temp(:, n) = sq_obs_op(:, n);
        K{o, n} = temp;
    end
end

end