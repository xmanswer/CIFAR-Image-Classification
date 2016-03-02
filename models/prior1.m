function [p] = prior1(yTrain, D)
c = 10;
n = length(yTrain); %get size of training set y
p = zeros([c 1]);
for i = 1:c
    %p(i) = sum(D' * (yTrain == (i-1))); %compute probablity for each class
    p(i) = sum((yTrain == (i-1)))/n; %compute probablity for each class
end
end

