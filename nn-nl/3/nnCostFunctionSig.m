function [L, gradh2o, gradh1h2, gradih1] = nnCostFunctionSig(X, y, thetaih1, thetah1h2, thetah2o, lambda)
    
    ntrain = length(y)
    nhidden2 = length(thetah2o(:, 1)) - 1
    nhidden1 = length(thetah1h2(:, 1)) - 1

    anon1 = sigmoid(X * thetaih1)
    
    a1 = [ones(ntrain, 1) anon1] % hidden layers values with bias term 
    
    anon2 = sigmoid(a1 * thetah1h2)
    
    a2 = [ones(ntrain, 1) anon2]
    
    outter = sigmoid(a2 * thetah2o)
    
    prob = exp(outter)
    
    norm = sum(prob')'
    
    prob(:, 1) = prob(:, 1) ./ norm
    prob(:, 2) = prob(:, 2) ./ norm % get output layer values and normalize by softmax function
    
    Y = zeros(ntrain, 2)
    Y(:, 1) = (y == 0)
    Y(:, 2) = (y == 1) % change Y into 2d vector
    

    L = -1 / ntrain * sum(sum(Y.*log(prob))) + lambda / 2 * (sum(sum(thetaih1.^2)) + sum(sum(thetah1h2.^2)) + sum(sum(thetah2o.^2)) - sum(thetaih1(1, :).^2) - sum(thetah1h2(1, :).^2) - sum(thetah2o(1, :).^2)) % loss function for training set
   
    
    deltaout = prob - Y
    deltaact2 = deltaout * thetah2o(2:nhidden2+1, :)' .* sigmoidG(anon2)
    deltaact1 = deltaact2 * thetah1h2(2:nhidden1+1, :)' .* sigmoidG(anon1)
    
    
    gradih1 = zeros(size(thetaih1))
    gradh1h2 = zeros(size(thetah1h2))
    gradh2o = zeros(size(thetah2o))
    
    
    gradh2o = gradh2o + a2' * deltaout / ntrain  + lambda * thetah2o
    gradh1h2 = gradh1h2 + a1' * deltaact2 / ntrain + lambda * thetah1h2
    gradih1 = gradih1 + X' * deltaact1 / ntrain + lambda * thetaih1 % calcualte gradient with respect to every weight components 
    
    gradh2o(1, :) = gradh2o(1, :) - lambda * thetah2o(1, :)
    gradh1h2(1, :) = gradh1h2(1, :) - lambda * thetah1h2(1, :)
    gradih1(1, :) = gradih1(1, :) - lambda * thetaih1(1, :) % remove bias contribution
    
%     aval = sigmoid(XVal * thetaih)
%     
%     aval = [ones(nval, 1) aval]
%         
%     outterVal = sigmoid(aval * thetaho)
%     
%     probVal = exp(outterVal)
%     
%     normval = sum(probVal')'
%     
%     probVal(:, 1) = probVal(:, 1) ./ normval
%     probVal(:, 2) = probVal(:, 2) ./ normval % output normalized values for validation sets
   
    
%    LVal = -1 / nval * sum(sum(YVal.*log(probVal))) + lambda / 2 * (sum(sum(thetaih.^2)) + sum(sum(thetaho.^2)) - sum(thetaih(1, :).^2) - sum(thetaho(1, :).^2)) % loss function for validation set
end