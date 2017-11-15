function [theta, J_history] = SgradientDescent(X, y, theta, alpha, num_iters)

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  for i = 1:m,
    delta = zeros(length(X(1, :)), 1);
    delta = delta + 1/m * (sigmoid(X(i,:) * theta) - y(i)) * X(i,:)';
    theta = theta - alpha * delta;
  end;
  J_history(iter) = costFunction(theta, X, y);

end

end