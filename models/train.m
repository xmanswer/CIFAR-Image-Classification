%% Main routine for training
function [Model] = train(X, Y)
%% pre processing sizes
% for i = 1:size(X,1)
%     X(i,:) = zscore(double(X(i,:)));
% end
X = double(X);
Y = double(Y)';
Ntrain = size(X, 1); %sample size
L = 10; %number of labels

%% parameters to be adjusted by cross-validation
sigma = 1.4; %sigma in guassian kernal
C = 100; %penlaty term
kernalType = 1; %0 for linear kernal, 1 for gaussian kernal

%% HOG.
CellSize = 8;
X_feat = zeros(size(X,1),(32/CellSize)*(32/CellSize)*31);
D = size(X_feat, 2);             % Number of Features. 
for i = 1:size(X,1)
    X_feat(i,:) = extract_feature(RowToImg(X(i,:)))';
end

%% do multi-class training
alpha = zeros(Ntrain, L);
w = zeros(D, L);
b = zeros(1, L);
for i = 1:L
    model = svmTrain(X_feat, Y, C, sigma, kernalType, i-1);
    alpha(:, i) = model.alpha;
    w(:, i) = model.w;
    b(1, i) = model.b;
end

%% passing parameters to output
Model.C = C;
Model.sigma = sigma;
Model.kernalType = kernalType;
Model.alpha = alpha;
Model.w = w;
Model.b = b;
Model.Xtrain = X_feat;
Model.Ytrain = Y;
end