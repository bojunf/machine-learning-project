function [Acc, AccTest] = Logistic(Trainfile, Testfile, alpha, niter, method)
    TrainData = importdata(Trainfile)
    ndata = size(TrainData(:, 1), 1)
    nfeat = size(TrainData(1, :), 2) - 1
    X = TrainData(:, 1: nfeat)
    Y = TrainData(:, nfeat+1)
    X = [ones(ndata, 1) X]
    
    TestData = importdata(Testfile)
    ntest = size(TestData(:, 1), 1)
    TestX = TestData(:, 1:nfeat)
    TestY = TestData(:, nfeat+1)
    TestX = [ones(ntest, 1) TestX]
    
    
%    alpha = 0.01
%    niter = 400
    
    theta = zeros(nfeat + 1, 1)
    
    if strcmp(method, 'GD')
        [theta, J_history] = gradientDescent(X, Y, theta, alpha, niter);
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
    
    save theta.txt theta -ascii
    save J-hist.txt J_history -ascii
    
    pred = round(sigmoid(theta'*X'))'
    
    Acc = sum(pred == Y) / ndata
    
    predTest = round(sigmoid(theta'*TestX'))'
    
    AccTest = sum(predTest == TestY) / ntest
    
%    disp(Acc)
    
end