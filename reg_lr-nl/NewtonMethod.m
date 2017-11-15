function [theta, J_history] = NewtonMethod(X, y, theta, alpha, num_iters)
    
m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  g = 1/m * X' * (sigmoid(X * theta) - y)
  S = 1/m * diag(sigmoid(X * theta))
  H = X' * S * X
    
  theta = theta - alpha * pinv(H) * g;
  J_history(iter) = costFunction(theta, X, y);

end

end