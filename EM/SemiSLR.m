function [] = SemiSLR(TrainfileL, TrainfileU, Testfile, alpha, niter, method, lambda, nem)
    TrainL = importdata(TrainfileL) % import training set
    nTrainL = size(TrainL(:, 1), 1) % number of training data
    ndim = size(TrainL(1, :), 2) - 1 % number of feature
    TrainLX = TrainL(:, 1:ndim)
    TrainLX = [ones(nTrainL,1) TrainLX]
    TrainLY = TrainL(:, ndim+1) % separate X and Y
    
    TrainU = importdata(TrainfileU) % import training set
    nTrainU = size(TrainU(:, 1), 1) % number of training data
    TrainUX = TrainU(:, 1:ndim)
    TrainUX = [ones(nTrainU,1) TrainUX]
    TrainUY = TrainU(:, ndim+1) % separate X and Y
    
    Test = importdata(Testfile) % import test set
    nTest = size(Test(:, 1), 1) % number of tests data    
    TestX = Test(:, 1:ndim)
    TestX = [ones(nTest,1) TestX]
    TestY = Test(:, ndim+1) % separate X and Y

    [theta, pred] = Logistic(TrainLX, TrainLY, alpha, niter, method, lambda)
    
    newTrainX = [TrainLX; TrainUX]
    
    for i = 1:nem
       TU = round(sigmoid(theta'*TrainUX'))'
       newY = [TrainLY; TU]
       [theta, pred] = Logistic(newTrainX, newY, alpha, niter, method, lambda)
    end
%    acc = sum(pred == TrainLY) / nTrainL
%     
    disp(size(TrainLY))
    disp(size(newY))
    accTest = sum(round(sigmoid(theta'*TestX'))' == TestY) / nTest
end