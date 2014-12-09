function [cost, grad] = softmaxCost(theta, numClasses, inputSize, ...
                                    lambda, data, labels)
%% Function
% softmaxCost computes the cost and gradient for softmax regression.
                                
%% Description of Inputs
% theta         - vector of model parameters theta - Kx(N+1)? x 1 
% numClasses    - the number of label classes - K
% inputSize     - the number of features - (N+1)?
% lambda        - weight decay parameter
% data          - design matrix - (N+1)? x M 
%                 each row (n,:) corresponds to a single feature
%                 each column (:,m) corresponds to a single example
% labels        - matrix of labels corresponding to the inputs - M x 1
%                 labels(m) is the class label for the mth example

%% Initialise Outputs
cost = 0;
thetagrad = zeros(numClasses, inputSize);

%% CODE

% Roll up theta parameters from vector into a K x (N+1) matrix
theta = reshape(theta, numClasses, inputSize);

% Number of training examples
m = size(data, 2);

% Ground Truth matrix
Y = full(sparse(labels, 1:m, 1));

% Hypothesis matrix
TX = theta * data;
TX = bsxfun(@minus, TX, max(TX, [], 1));  % Prevents overflow
Exp_TX = exp(TX);
H = bsxfun(@rdivide, Exp_TX, sum(Exp_TX, 1));

% Cost
cost = (-1/m) * sum(sum(Y .* log(H))) +...
       (lambda/2) * sum(sum(theta .^2));

% Gradient
thetagrad = (-1/m) * (Y - H) * data' +...
            lambda * theta;

% Unroll the gradient matrices into a vector for minFunc
grad = thetagrad(:);

end

