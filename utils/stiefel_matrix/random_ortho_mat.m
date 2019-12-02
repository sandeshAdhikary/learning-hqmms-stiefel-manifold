function [mat] = random_ortho_mat(nrows, ncols)
    % Create a random orthogonal matrix of dimensions nrows x ncols
    
    mat = complex(rand(nrows,ncols),rand(nrows,ncols));
    
    try 
        mat = nearest_orthonorm(mat,'singVals');
    catch
        fprintf('Could not use SVD method, trying Gram Schmidt\n');
        mat = nearest_orthonorm(mat,'gramSchmidt');
    end

end