function [M, V] = likelihood1(xTrain, yTrain, D)
c = 10;
[n f] = size(xTrain); %get size of trainning set n and feature f

for i = 1:c %for each class
    %create a vector with 1 in the position of this class (n x 1)
    labelVector = (yTrain == (i - 1));
    %compute # of trainning samples in this class
    sumY = sum(labelVector);
    %compute mean for this class for every feature (f x 1)
    %M(:,i) = xTrain' * (labelVector .* D) / sum(labelVector .* D);
    M(:,i) = xTrain' * (labelVector) / sumY;
    %create repeated matrix for mean on each tranning sample
    mu = repmat(M(:,i), 1, n);
%     DD = repmat(D', f, 1);
    %compute variance for this class for every feature (f x 1)
    %V(:, i) = (xTrain' - mu) .^ 2 * (labelVector .* D) / sum(labelVector .* D);
    V(:, i) = (xTrain' - mu) .^ 2 * (labelVector) / sumY;
end
end
