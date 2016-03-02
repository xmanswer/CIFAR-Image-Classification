function [Y] = classify2(Model, X)
L = 10;
alpha = Model.alpha;
model = Model.model;
iter = size(alpha, 1);
X = double(X);
for i = 1:size(X,1)
    X(i,:) = zscore(double(X(i,:)));
end

%% HOG.
CellSize = 4;
X_feat = zeros(size(X,1),(32/CellSize)*(32/CellSize)*31);
% D = size(X_feat, 2);             % Number of Features. 
for i = 1:size(X,1)
    X_feat(i,:) = extract_feature4(RowToImg(X(i,:)))';
end

%%
N = size(X_feat, 1);
Y_classify = zeros(N, iter);
Y_vote = zeros(N, L);


for i = 1 : iter
    p = model(i).p;
    M = model(i).M;
    V = model(i).V;
    Y_classify(:,i) = naiveBayesClassify2(X_feat, M, V, p);
end

for i = 1:L
    Y_vote(:, i) = (Y_classify == (i-1)) * alpha;
end

[~, Y] = max(Y_vote,[],2);
Y = Y - 1;
%Y = naiveBayesClassify(X, Model.M, Model.V, Model.p);
end