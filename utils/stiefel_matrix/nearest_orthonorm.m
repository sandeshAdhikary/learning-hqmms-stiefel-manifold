function [orthoNormA] = nearest_orthonorm(A, strategy)
% Return the nearest orthogonal matrix to A

    if strcmp(strategy,'singVals')
        % % Option 1: Set all Singular Values to 1
        [leftU,Sigma,rightV] = svd(A);
        Sigma = eye(size(Sigma,1),size(Sigma,2));
        orthoA = leftU*Sigma*(ctranspose(rightV));
        colNorms = sqrt(sum(arrayfun(@conjProd,orthoA),1));
        orthoNormA = bsxfun(@rdivide,orthoA,colNorms);
        
    elseif strcmp(strategy,'gramSchmidt')
        % % Option 2: Gram Schmidt
        orthoNormA = gramSchmidt(A);
    else
        error('Unknown orthogonalization strategy')
    end
    
end

function ip = conjProd(a)
    ip = conj(a)*a;
end

function orthoNormMat = gramSchmidt(A)
    % Given a matrix A with linearly independent columns, 
    % Returns the matrix expressed in orthonormal basis

    [~,n] = size(A);

    if rank(A) ~= n
        % Make sure columns are linearly independent
        error('Matrix is not full rank')
    end

    orthoNormMat = A(:,1);
    orthoNormMat = orthoNormMat/sqrt((ctranspose(orthoNormMat)*orthoNormMat)); 
    % ^ normalize
    for k = 2:n %Loop through columns
        z = 0; %Variable for residues to delete from column
        for j = 1:k-1 %Loop through cols already ortho-ed
            p = orthoNormMat(:,j);
            residue =  (ctranspose(p)*A(:,k)/(ctranspose(p)*p))*p;
            z = z + residue; %add up the residues
        end
        newCol = A(:,k)-z; %subtract the residue
        newCol = newCol/sqrt((ctranspose(newCol)*newCol)); %Normalize
        orthoNormMat = [orthoNormMat newCol]; %append to matrix
    end


end