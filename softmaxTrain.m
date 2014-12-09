function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda,...
                                       inputData, labels, options)
%% Function
% softmaxTrain trains a softmax model with the given parameters on
% the given data.
% The output softmax.Model contains softmaxOptTheta (vector containing
% the trained model parameters), inputSize & numClasses

%% Inputs
% inputSize     - the number of features (N+1)?
% numClasses    - the number of label classes (K)
% lambda        - weight decay parameter
% inputData     - the (N+1) x M design matrix
%                 each row (n,:) corresponds to a single feature
%                 each column (:,m) corresponds to a single example
% labels        - M x 1 matrix of labels corresponding to the inputs 
%                 labels(m) is the class label for the mth example
% options       - options (optional) for the optimisation algorithm

%% Code

% Initialize parameters theta
% Note that theta parameters are unrolled into a single vector as the
% optimisation algorithm expects this.
theta = 0.005 * randn(numClasses * inputSize, 1);

% Set some optimisation options
% If 'options' is not given as an input -> create it
if ~exist('options', 'var')
    options = struct;
end
% If max # of iterations is not set, set to 400
if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

% Use minFunc to minimize the function and find optimised theta
% minFunc is an unconstrained optimiser (similar to fminunc)

addpath minFunc/
options.Method = 'lbfgs';           % Using L-BFGS to optimise
minFuncOptions.display = 'on';

[softmaxOptTheta, cost] = minFunc( @(p) ...
                            softmaxCost(p, numClasses, inputSize, ...
                                        lambda, inputData, labels), ...                                   
                          theta, options);

% Fold softmaxOptTheta into a nicer format
% Reshape theta to a K x (N+1)? matrix
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
                          
end                          
