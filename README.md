# Dependencies:
1. MATLAB2018b with the QETLAB (for quantum density operators) and the Bioinformatics toolbox (for cross validation tools).
2. Python 3.5 with autograd, and other standard packages in Anaconda.

# Working with Matlab and Python:
We use python for gradient calculations and matlab for all other computations. Data flow between these platforms was tested for MATLAB2018b and Python3.5; other combinations may result in errors due to a lack of cross-platform integration. You will need to set up the Python3.5 interpreter path in matlab as shown here: https://www.mathworks.com/help/matlab/ref/pyversion.html.

# Running Scripts:
Add the top cosm folder and all subfolders to MATLAB's path, and run experiments from the top folder. Scripts related to the three datasets in the paper are organized into individual folders inside /evaluation_scripts. Inside each folder, run the *_exp.m file, set up the model hyperparameters, and run experiment. Results and learned models will be saved in the corresponding results folder. The tuned hyperparameters presented in the paper can be found in the respective 'tuned_hyperparameters' folders. Note that we used tuned parameters when they yielded better results than the default ones provided in the exp.m files.

These exp.m files will learn an HQMM using COSM, and also an HMM using EM for reference.

We describe the main files below:

## Learning Scripts:

*learn_qgm.m*: The learn_qgm() function takes as input some data, an initial guess for learned parameters. It runs COSM optimization and returns the learned parameter (K_best) and an array with the training trace. Training specifics (including hyperparameters) are passed as fields of a 'params' struct. The validation dataset is passed as an optional input. Note that the code currently has references to an alternate model class calledn 'qnb': this is not relevant to the current paper, and is a placeholder for future work.

Note: In the rare occasion where intialized matrices have exact zeros, add noise to kraus ops/density matrix to avoid numerical issues.

*learn_hmm.m*: The learn_hmm() function takes as input some training data, and model specifics in the struct 'params'. It learns an HMM model using the EM algorithm for comparison with COSM. Input requirements are similar to learn_qgm.


## Utilities Scripts:
### General
Scripts to compute loss and the gradient (in python).

### HQMM
Scripts specific to HQMM e.g. computing loss, evaluating performance, etc.

### Stiefel Matrix
Scripts specific to Stiefel matrices e.g. reshapes, generating random guesses, etc.

# Data:
The 3 datasets used in the paper are in the /data folder.

    
