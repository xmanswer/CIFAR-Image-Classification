%%
function [Yout] = svmClassify(X, targetClass, Model)
kernalType = Model.kernalType;
C = Model.C;
sigma = Model.sigma;
alpha = Model.alpha;
alpha = alpha(:, targetClass);
w = Model.w;
w = w(:, targetClass);
b = Model.b;
b = b(1, targetClass);

if kernalType == 0 %linear kernal
    prediction = X * w + b; 
else %gaussian kernal
    supVec = (alpha > 0) & (alpha < C);
    Xtrain = Model.Xtrain;
    Xtrain = Xtrain(supVec, :);
    Ytrain(Model.Ytrain == (targetClass-1)) = 1;
    Ytrain(Model.Ytrain ~= (targetClass-1)) = -1;
    Ytrain = Ytrain(supVec);
    alpha = alpha(supVec);
    alpha = alpha';
    
    X1 = sum(X .^ 2, 2);
    X2 = sum(Xtrain .^ 2, 2)';
    X_final = bsxfun(@plus, X2, - 2 * X * Xtrain');
    X_final = bsxfun(@plus, X1, X_final);
    Kernal = exp(-1/(2 * sigma^2)) .^ (X_final);
    Ytrain = repmat(Ytrain, size(Kernal, 1), 1);
    alpha = repmat(alpha, size(Kernal, 1), 1);
    Kernal = Kernal .* Ytrain .* alpha;
    prediction = sum(Kernal, 2);
end

Yout = prediction + b;
end