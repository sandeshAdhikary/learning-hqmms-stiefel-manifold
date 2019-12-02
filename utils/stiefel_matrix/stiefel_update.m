function [K_mat] = stiefel_update(K_mat, G_mat, step_size)
    % Compute the S update using the Stiefel optimizations in the paper
    % Output: New S after update
    
    % More efficient with smaller inversions
    U = [G_mat K_mat];
    V = [K_mat -G_mat];
    invTerm = ctranspose(V)*U;
    invTerm = eye(size(invTerm, 1)) + (step_size*invTerm/2);
%     update = step_size*U*invTerm\(ctranspose(V)*K_mat);
    A = step_size*U*inv(invTerm);
    B = ctranspose(V)*K_mat;
    update = A*B;
    
    K_mat = K_mat - update;
    
    % Projection (only needed if initial S is not on the manifold)
    ortho_counter = 1;
    ortho_error = norm(ctranspose(K_mat)*K_mat - eye(size(K_mat, 2)));
    while ortho_error >= 10e-8
        fprintf('[Retraction] Orthogonality Error %e: Projecting...\n', ...
                                                               ortho_error)
        K_mat = nearest_orthonorm(K_mat, 'singVals');
        ortho_error = norm(ctranspose(K_mat)*K_mat - eye(size(K_mat,2)));
        ortho_counter = ortho_counter + 1;
        if ortho_counter > 10
            error('Could not make K_mat orthogonal with 10 projections')
        end
    end
    
end
