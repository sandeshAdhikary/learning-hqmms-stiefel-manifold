function [K_tensor] = mat_to_tensor(K_mat, dim1, dim2, dim3, dim4)
% Given a Stiefel matrix with stacked Kraus operators
% Return the corresponding tensor

    K_tensor = permute(reshape(K_mat, [dim3 dim2 dim1 dim4]), [3 2 1 4]);
    
end