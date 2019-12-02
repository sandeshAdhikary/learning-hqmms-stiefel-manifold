function [K_mat] = tensor_to_mat(K_tensor, stiefel_nrows, stiefel_ncols)
% Given a tensor of Kraus operators, convert it to a Stiefel matrix

    K_mat = transpose(reshape(permute(K_tensor, [4 3 2 1]), ...
                                           [stiefel_ncols stiefel_nrows]));
end