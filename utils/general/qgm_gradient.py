import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
from autograd.scipy.misc import  logsumexp

def logdotexp(logA, logB):
    """
        Given the log of two matrices A and B as logA and logB, carries out A * B in log space
    """
    maxA = np.max(logA)
    maxB = np.max(logB)
    if np.abs(maxA) == np.Inf:
        sortA = np.sort(logA.reshape(-1))
        if np.sum(np.abs(sortA) != np.Inf) == 0:
            C = np.zeros([logA.shape[0], logB.shape[1]], dtype='complex128') - np.Inf
            return C
        else:
            maxA = sortA[np.sum(np.abs(sortA) != np.Inf) - 1]

    if np.abs(maxB) == np.Inf:
        sortB = np.sort(logB.reshape(-1))
        if np.sum(np.abs(sortB) != np.Inf) == 0:
            C = np.zeros([logA.shape[0], logB.shape[1]], dtype='complex128') - np.Inf;
            return C
        else:
            maxB = sortB[np.sum(np.abs(sortB) != np.Inf)]
    
    expA = np.exp(logA - maxA)
    expB = np.exp(logB - maxB)

    C = np.log(np.dot(expA, expB)) + maxA + maxB

    return C


def logaddexp(logA, logB):
    """
        Given the log of two matrices A and B as logA and logB, carries out A + B in log space
    """

    maxN = np.maximum(logA, logB)
    minN = np.minimum(logA, logB)

    C = np.log1p(np.exp(minN - maxN)) + maxN 

    return C


def get_hqmm_gradient(K_real, K_imag, rho_real, rho_imag, batch, burn_in, strategy):
    
    K = np.array(K_real) + np.complex(0,1)*np.array(K_imag)
    rho = np.array(rho_real) + np.complex(0,1)*np.array(rho_imag)
    batch = np.array(batch)
    burn_in = int(burn_in)
    if len(batch.shape) in [0,1]:
        batch = batch.reshape(1,-1)

    def log_loss(K_conj):
        """
            K is a tensor of CONJUGATE Kraus Operators of dim s x w x n x n
            s: output_dim
            w: ops_per_output
            n: state_dim
        """
        total_loss = 0.0
        
        # Iterate over each sequence in batch
        for i in range(batch.shape[0]):
            seq = batch[i]

            rho_new = np.log(rho.copy())

            # burn in
            for b in range(burn_in):
                temp_rho = np.zeros([K_conj.shape[1], K_conj.shape[2], K_conj.shape[3]], dtype='complex128')
                for w in range(K_conj.shape[1]):
                    temp_rho[w, :, :] = np.dot(np.dot(K[int(seq[b])-1, w, :, :], rho_new), np.conjugate(K[int(seq[b])-1, w, :, :]).T)
                rho_new = np.sum(temp_rho, 0)
                rho_new = rho_new/np.trace(rho_new)

            # Compute likelihood for the sequence            
            for s in seq[burn_in:]:
                rho_sum = logdotexp( logdotexp( np.log(np.conjugate(K_conj[int(s)-1, 0, :, :])), rho_new ), np.log(K_conj[int(s)-1, 0, :, :].T))
                for w in range(1, K_conj.shape[1]):
                    # subtract 1 to adjust for MATLAB indexing
                    rho_sum = logaddexp(rho_sum, logdotexp( logdotexp( np.log(np.conjugate(K_conj[int(s)-1, w, :, :])), rho_new ), np.log(K_conj[int(s)-1, w, :, :].T)))

                rho_new = rho_sum

            total_loss += np.real(logsumexp(np.diag(rho_new)))

        return -total_loss/batch.shape[0]


    def loss(K_conj):
        """
            K is a tensor of CONJUGATE Kraus Operators of dim s x w x n x n
            s: output_dim
            w: ops_per_output
            n: state_dim
        """
        total_loss = 0.0
        
        # Iterate over each sequence in batch
        for i in range(batch.shape[0]):
            seq = batch[i]
            rho_new = rho.copy()
            # burn in
            for b in range(burn_in):
                temp_rho = np.zeros([K_conj.shape[1], K_conj.shape[2], K_conj.shape[3]], dtype='complex128')
                for w in range(K_conj.shape[1]):
                    temp_rho[w, :, :] = np.dot(np.dot(K[int(seq[b])-1, w, :, :], rho_new), np.conjugate(K[int(seq[b])-1, w, :, :]).T)
                rho_new = np.sum(temp_rho, 0)
                rho_new = rho_new/np.trace(rho_new)

            # Compute likelihood for the sequence
            for s in seq[burn_in:]:
                rho_sum = np.zeros([K_conj.shape[2], K_conj.shape[2]],  dtype='complex128')
                for w in range(K.shape[1]):
                    # subtract 1 to adjust for MATLAB indexing
                    rho_sum += np.dot( np.dot( np.conjugate(K_conj[int(s)-1, w, :, :]), rho_new ), K_conj[int(s)-1, w, :, :].T)

                rho_new = rho_sum

            total_loss += np.log(np.real(np.trace(rho_new)))

        return -total_loss/batch.shape[0]


    if strategy == 'logloss':
        grad_fn = grad(log_loss)
        gradient = grad_fn(np.conjugate(K))
    elif strategy == 'loss':
        grad_fn = grad(loss)
        gradient = grad_fn(np.conjugate(K))
    else:
        raise Exception('Unknown Loss Strategy')    

    return np.real(gradient),np.imag(gradient)


def get_qnb_gradient(labels, feats_matrix, K_real, K_imag, strategy):
        
    K = np.array(K_real) + np.complex(0,1)*np.array(K_imag)
    feats_matrix = np.array(feats_matrix, dtype = 'int32')
    labels = np.array(labels,dtype = 'int32')
    
    def logloss(K_conj):
        """
            K is a tensor of CONJUGATE Kraus Operators of dim s x y x x x x
            s: dim of features
            y: number of features
            x: number of labels
        """
        total_loss = 0.0
        
        # Iterate over each sequence in batch
        for i in range(labels.shape[0]):
            features = feats_matrix[i, :]
            label = labels[i] - 1

            # Compute likelihood of the label generating the given features
            conjKrausProduct = np.log(K_conj[features[0]-1, 0, :, :])
            for s in range(1, features.shape[0]):
                conjKrausProduct = logdotexp(np.log(K_conj[features[s]-1, s, :, :]), conjKrausProduct)
            
            eta = np.zeros([K_conj.shape[3],K_conj.shape[3]], dtype='complex128')
            eta[label,label] = 1

            prod1 = logdotexp(np.conjugate(conjKrausProduct), np.log(eta))
            prod2 = logdotexp(prod1, conjKrausProduct.T)
            total_loss += np.real(logsumexp(np.diag(prod2)))
            
            # total_loss += np.real(np.trace(np.kron(np.conjugate(conjKrausProduct)[:, label], conjKrausProduct.T[:, label]).reshape(K_conj.shape[2], K_conj.shape[3])))

        return -total_loss/labels.shape[0]

    def losswlog(K_conj):
        """
            K is a tensor of CONJUGATE Kraus Operators of dim s x y x x x x
            s: dim of features
            y: number of features
            x: number of labels
        """
        total_loss = 0.0
        
        # Iterate over each sequence in batch
        for i in range(labels.shape[0]):
            features = feats_matrix[i, :]
            label = labels[i] - 1

            # Compute likelihood of the label generating the given features
            conjKrausProduct = K_conj[features[0]-1, 0, :, :]
            for s in range(1, features.shape[0]):
                conjKrausProduct = np.dot(K_conj[features[s]-1, s, :, :], conjKrausProduct)
            
            eta = np.zeros([K_conj.shape[3],K_conj.shape[3]], dtype='complex128')
            eta[label,label] = 1

            prod1 = np.dot(np.conjugate(conjKrausProduct), eta)
            prod2 = np.dot(prod1, conjKrausProduct.T)
            total_loss += np.log(np.real(np.trace(prod2)))
            
            # total_loss += np.real(np.trace(np.kron(np.conjugate(conjKrausProduct)[:, label], conjKrausProduct.T[:, label]).reshape(K_conj.shape[2], K_conj.shape[3])))

        return -total_loss/labels.shape[0]

    if strategy == 'losswlog':
        grad_fn = grad(losswlog)
        gradient = grad_fn(np.conjugate(K))
    else:
        grad_fn = grad(logloss)
        gradient = grad_fn(np.conjugate(K))
    

    return [np.real(gradient),  np.imag(gradient)]