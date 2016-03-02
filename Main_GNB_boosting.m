%% Test svmTrain
clear;
X = zeros(5000,3072);
Y = zeros(5000,1);
load('small_data_batch_1.mat')
for i = 1:1000
    X(i,:) = data(i,:);
end
Y(1:1000) = labels;
load('small_data_batch_2.mat')
for i = 1:1000
    X(1000+i,:) = data(i,:);
end
Y(1001:2000) = labels;
load('small_data_batch_3.mat')
for i = 1:1000
    X(2000+i,:) = data(i,:);
end
Y(2001:3000) = double(labels);
load('small_data_batch_4.mat')
for i = 1:1000
    X(3000+i,:) = data(i,:);
end
Y(3001:4000) = labels;
load('small_data_batch_5.mat')
for i = 1:1000
    X(4000+i,:) = data(i,:);
end
Y(4001:5000) = labels;

%% Train and Classify. 
         
% tic
% Model2 = train2(X,Y);
% toc
load('Model2.mat');
load('data_batch_1.mat');
X_test = data(1:1000,:);
Y_true = double(labels(1:1000));
Ytest = classify2(Model2,X_test);
TestAcc = sum(Ytest == Y_true)/length(Y_true);
