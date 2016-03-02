%%
function [Y] = classify(Model, X)
X = double(X);
L = 10; %number of labels
NTest = size(X, 1);
YTest = zeros(NTest, L);
%% Feature Selection. 
CellSize = 8;
X_feat = zeros(size(X,1),(32/CellSize)*(32/CellSize)*31);

for i = 1:size(X,1)
    X_feat(i,:) = extract_feature(RowToImg(zscore(X(i,:))))';
end

for j = 1:L
    YTest(:,j) = svmClassify(X_feat, j, Model);
end

[~, Y] = max(YTest');
Y = Y - 1;



end