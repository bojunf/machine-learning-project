function [theta, weightErr, pred] = Logistic(TrainData, alpha, niter, method, D, lambda)
%    TrainData = importdata('norm_train.txt')
    ndata = size(TrainData(:, 1), 1)
    nfeat = size(TrainData(1, :), 2) - 1
    X = TrainData(:, 1: nfeat)
    Y = TrainData(:, nfeat+1)
    X = [ones(ndata, 1) X]
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
    Err = (pred ~= Y)
    acc = sum(1-Err) / ndata
    weightErr = D' * Err
    
end