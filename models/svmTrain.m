%% one versus all training, train parameters for given targetClass
function [Model] = svmTrain(X, Y, C, sigma, kernalType, targetClass)
%% pre processing labels and sizes
Y_binary(Y == targetClass) = 1; %Ntrain x 1
Y_binary(Y ~= targetClass) = -1;
Y_binary = Y_binary';
Ntrain = size(X, 1); %sample size
D = size(X, 2); %4 x K features from feature selection

%% compute kernal(i, j), Ntrain x Ntrain matrix
Kernal = zeros(Ntrain,Ntrain);
if kernalType == 0 %linear kernal
    Kernal = X * X';
else %gaussian kernal
    %faster way to do exp(-(Xi-Xj)^2/(2*sigma^2))
    X_sumFeature = sum(X .^ 2, 2);
    X_sumEachOther = bsxfun(@plus, X_sumFeature, X_sumFeature');
    X_final = bsxfun(@plus, -2 * (X * X'), X_sumEachOther);
    Kernal = exp(-1/(2 * sigma^2)) .^ (X_final);   
%     for i = 1:Ntrain
%         for j = 1:Ntrain
%             Kernal(i,j) = exp(-1/(2 * sigma^2)*(norm(X(i,:)-X(j,:))^2));
%         end
%     end
end

%% use qp solver to solve alpha for minimizing
% -sum_i(alphai) + 1/2 * sum_i_j(alphai * alphaj * Yi * Yj * Kernal_i_j)
% w.r.t. alpha, s.t. sum_i(alphai * Yi) = 0, C > alphai >= 0

H = (Y_binary * Y_binary') .* Kernal; % Hessian matrix Ntrain x Ntrain
f = -1 * ones(Ntrain, 1);
A = [];
b = [];
Aeq = Y_binary';
beq = 0;
lb = zeros(Ntrain,1);
ub = C * ones(Ntrain, 1);
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub); %Ntrain x 1 vector

%% post processing parameters alpha, w, b
%approximate out points don't belong to support vectors
alpha(alpha < 1e-4) = 0;
alpha(alpha > (1-1e-4) * C) = 0;
%compute w, a D x 1 vector, useful if it is linear kernal
w = ((alpha .* Y_binary)' * X)';
%Ntrain x 1 vector indicating support vectors
supVec = (alpha > 0) & (alpha < C); 
%get bias term by averaging b calculated from support vectors
b = supVec' * (Y_binary - ((alpha .* Y_binary)' * Kernal)') / sum(supVec);

%% passing input parameters to output
Model.C = C;
Model.sigma = sigma;
Model.kernalType = kernalType;
Model.targetClass = targetClass;

%% passing trained parameters to output
Model.alpha = alpha;
Model.w = w;
Model.b = b;
end