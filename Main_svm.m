%% Test svmTrain
clear;
% C_rng = 10;
% sigma_rng = 25.6;
% C_rng = 100;
% sigma_rng = 1.4;
% Train_acc = zeros(length(C_rng),length(sigma_rng));
% Test_acc = zeros(length(C_rng),length(sigma_rng));
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

%% HoG Selection
% CellSize = 8;
% X_train_pre = X(1:1000,:);
% X_train = zeros(size(X_train_pre,1),(32/CellSize)*(32/CellSize)*31);
% Y_train = Y(1:1000);
% X_test_pre = X(9001:10000,:);
% X_test = zeros(size(X_test_pre,1),(32/CellSize)*(32/CellSize)*31);
% Y_true = double(labels(9001:10000));
% for i = 1:size(X_train_pre,1)
%     X_train(i,:) = extract_feature(RowToImg(X_train_pre(i,:)))';
% end
% for i = 1:size(X_test_pre,1)
%     X_test(i,:) = extract_feature(RowToImg(X_test_pre(i,:)))';
% end

%% PCA Selection
% NOfF = 100;
% X_train = X(1:1000,:);
% [X_train, coeff] = PCA_DimRed(X_train,NOfF);
% Y_train = Y(1:1000);
% X_test = X(9001:10000,:);
% X_test = X_test * coeff;
% X_test = X_test(:,1:NOfF);
% Y_true = double(labels(9001:10000));

%% Train and Classify. 
         
tic
Model = train(X,Y);
toc
% load('Model.mat');
load('data_batch_1.mat');
X_test = double(data(1:1000,:));
Y_true = double(labels(1:1000));
Ytest = classify(Model,X_test)';
TestAcc = sum(Ytest == Y_true)/length(Y_true);

% tic
% Ytest_train = classify(Model,X_train)';
% toc
% TrainAcc = sum(Ytest_train == Y_train)/length(Y_train);
% Train_acc(C_cnt,sigma_cnt) = TrainAcc;       
% Ytest = classify(Model,X_test)';
% TestAcc = sum(Ytest == Y_true)/length(Y_true);
% Test_acc(C_cnt,sigma_cnt) = TestAcc;
% 
% save('ACC.mat','Train_acc','Test_acc','C_rng','sigma_rng');