function [theta, pred] = Logistic(X, Y, alpha, niter, method, lambda)
%    TrainData = importdata('norm_train.txt')
    ndata = size(X(:, 1), 1)
    nfeat = size(X(1, :), 2) - 1
%    X = TrainData(:, 2: nfeat+1)
%    Y = TrainData(:, nfeat+1)
%    X = [ones(ndata, 1) X]
%    alpha = 0.01
%    niter = 400
%    D = ones(ndata, 1) / ndata
    
    theta = zeros(nfeat+1, 1)
    
    if strcmp(method, 'GD')
        [theta, J_history] = gradientDescent(X, Y, theta, alpha, niter, lambda);
    elseif strcmp(method, 'SG')
        [theta, J_history] = SgradientDescent(X, Y, theta, alpha, niter);
    elseif strcmp(method, 'NM')  
        [theta, J_history] = NewtonMethod(X, Y, theta, alpha, niter);
    elseif strcmp(method, 'FM')
        options = optimset('GradObj', 'on');
        [theta, J_history] = fminunc(@(t)(costFunctionFminunc(t, X, Y)), theta, options);
    else
        disp('Wrong input method.');
        return
    end
    

%     figure;
%     plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%     xlabel('Number of iterations');
%     ylabel('Cost J');
    
%     save theta.txt theta -ascii
%     save J-hist.txt J_history -ascii
    
    pred = round(sigmoid(theta'*X'))'
    acc = sum(pred == Y) / ndata
    
end