function J = costFunction(theta, X, y)
    m = length(y) % number of data
    J = 0 
    grad = zeros(size(theta)) % initialize cost function and gradient
    
    J = J + 1/m * (-sum(y.*log(sigmoid(X*theta))) - sum((1-y).*log(1 - sigmoid(X*theta))))
%    grad = grad + 1/m * X' * (sigmoid(X*theta)-y) % updat cost function and gradient
    
end