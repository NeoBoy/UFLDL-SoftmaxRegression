clear all; clc;

%% STEP 0: Set parameters

%  Initialise parameters used for tuning the model.
lambda = 1e-4;                          % Weight decay parameter

%%====================================================================
%% STEP 1: Load input and output data

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

numClasses = size(unique(labels), 1);   % Number of classes
labels(labels==0) = 10;                 % Remap 0 to 10

inputData = images;
numFeatures = size(inputData, 1);       % Number of features

%%====================================================================
%% STEP 2: Learning parameters

%  Train softmax regression model using softmaxTrain.m
%  - uses softmaxCost.m and minFunc

options.maxIter = 100;
softmaxModel = softmaxTrain(numFeatures, numClasses, lambda, ...
                            inputData, labels, options);

%%====================================================================
%% STEP 3: Testing

%  Test model predictions against the test images.

images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;

% Use function softmaxPredict.m
[prob, pred] = softmaxPredict(softmaxModel, inputData);

% Accuracy is the proportion of correctly classified images
acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% Analyse probability next ...
% [pred(1:50)' labels(1:50) pred(1:50)'~=labels(1:50) prob(1:50)']