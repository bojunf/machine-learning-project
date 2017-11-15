function [L, gradho, gradih] = nnCostFunctionSig(X, y, thetaih, thetaho, lambda)
    
    ntrain = length(y)
    nhidden = length(thetaho(:, 1)) - 1

    anon = sigmoid(X * thetaih)
    
    a = [ones(ntrain, 1) anon] % hidden layers values with bias term 
    
    outter = sigmoid(a * thetaho)
    
    prob = exp(outter)
    
    norm = sum(prob')'
    
    prob(:, 1) = prob(:, 1) ./ norm
    prob(:, 2) = prob(:, 2) ./ norm % get output layer values and normalize by softmax function
    
    Y = zeros(ntrain, 2)
    Y(:, 1) = (y == 0)
    Y(:, 2) = (y == 1) % change Y into 2d vector
    

    L = -1 / ntrain * sum(sum(Y.*log(prob))) + lambda / 2 * (sum(sum(thetaih.^2)) + sum(sum(thetaho.^2)) - sum(thetaih(1, :).^2) - sum(thetaho(1, :).^2)) % loss function for training set
   
    
    deltaout = prob - Y
    deltaact = deltaout * thetaho(2:nhidden+1, :)' .* sigmoidG(anon)
    
    gradih = zeros(size(thetaih))
    gradho = zeros(size(thetaho))
    
    
    gradho = gradho + a' * deltaout / ntrain  + lambda * thetaho
    gradih = gradih + X' * deltaact / ntrain + lambda * thetaih % calcualte gradient with respect to every weight components 
    
    gradho(1, :) = gradho(1, :) - lambda * thetaho(1, :)
    gradih(1, :) = gradih(1, :) - lambda * thetaih(1, :) % remove bias contribution
    
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