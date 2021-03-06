function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters, lambda)

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  delta = zeros(length(X(1, :)), 1);
  for i = 1:m,
    delta = delta + 1/m * (sigmoid(X(i,:) * theta) - y(i)) * X(i,:)';
  end;
  theta = theta - alpha * (delta + lambda * (theta - theta(1, :)));
  J_history(iter) = costFunction(theta, X, y, lambda);

end

end