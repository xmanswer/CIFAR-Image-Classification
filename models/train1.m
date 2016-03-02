function [Model] = train1(X, Y)
X = double(X);
for i = 1:size(X,1)
    X(i,:) = zscore(double(X(i,:)));
end
Y = double(Y);

%% HOG.
CellSize = 4;
X_feat = zeros(size(X,1),(32/CellSize)*(32/CellSize)*31);
% D = size(X_feat, 2);             % Number of Features. 
for i = 1:size(X,1)
    X_feat(i,:) = extract_feature4(RowToImg(X(i,:)))';
end

%%
N = size(X_feat, 1);
D = 1/N * ones(N, 1); % N x 1 matrix for weights of samples
% iter = 5;
% alpha = zeros(iter, 1);

% for i = 1:iter
%     pt = prior(Y, D);
%     [Mt, Vt] = likelihood(X, Y, D);
%     model(i).p = pt;
%     model(i).M = Mt;
%     model(i).V = Vt;
%     Y_classify = naiveBayesClassify(X, Mt, Vt, pt);
%     eps = D' * (Y_classify ~= Y) % 1xN .* Nx1
%     sum(Y_classify == Y)/length(Y)
%     alpha(i) = 1/2 * log((1 - eps) / eps)
%     D = D .* exp(- alpha(i) * (Y_classify == Y));
%     D = D / sum(D);
% end
% Model.alpha = alpha;
% Model.model = model;
p = prior1(Y, D);
[M, V] = likelihood1(X_feat, Y, D);
Model.p = p;
Model.M = M;
Model.V = V;
end