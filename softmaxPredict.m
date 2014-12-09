function [prob, pred] = softmaxPredict(softmaxModel, data)
%% Function
% pred makes predictions on input data using the trained softmax
% model
                                
%% Description of Inputs
% softmaxModel - model trained using softmaxTrain
% data         - the N x M input matrix
%                each column data(:, i) corresponds to a single
%                test set

%% Description of Output
% pred         - prediction matrix 
%                where pred(i) is argmax_c P(y(c) | x(i)).

%% Code
% Optimised parameters theta matrix - K x (N+1)?
theta = softmaxModel.optTheta;

% Initialise output
pred = zeros(1, size(data, 2));

% Easier if don't need probability
% p = theta*data;
% [~, pred] = max(p, [], 1);

TX = theta * data;
TX = bsxfun(@minus, TX, max(TX, [], 1));  % Prevents overflow
Exp_TX = exp(TX);
H = bsxfun(@rdivide, Exp_TX, sum(Exp_TX, 1));
[prob, pred] = max(H, [], 1);

end

