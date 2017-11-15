function [] = Adaboost(TrainName, TestName, T, alpha, niter, method, lambda)
    Train = importdata(TrainName) % import training set
    nTrain = size(Train(:, 1), 1) % number of training data
    ndim = size(Train(1, :), 2) - 1 % number of feature
    TrainX = Train(:, 1:ndim)
    TrainX = [ones(nTrain,1) TrainX]
    TrainY = Train(:, ndim+1) % separate X and Y
    TrainYref = TrainY * 2 - 1
    Test = importdata(TestName) % import test set
    nTest = size(Test(:, 1), 1) % number of tests data    
    TestX = Test(:, 1:ndim)
    TestX = [ones(nTest,1) TestX]
    TestY = Test(:, ndim+1) % separate X and Y
%    TestY = TestY * 2 - 1
    D = ones(nTrain, 1) / nTrain % initialize D

    weight = zeros(T, 1)
    classifier = zeros(ndim+1, T)
    TrainClass = zeros(nTrain, T)
    TestClass = zeros(nTest, T)
    accTrain = zeros(T, 1)
    accTest = zeros(T, 1)
    Dhis = zeros(nTrain, T)
    
    for t = 1:T
        [theta, weightErr, pred] = Logistic(Train, alpha, niter, method, D, lambda) % prediction based on current D
        weight(t) = 0.5 * log((1 - weightErr)/ max(weightErr, 1e-16)) % save weight of this stump
        classifier(:, t) = theta % save stump of this t
        predref = pred * 2 - 1
        dUpdata = exp(-weight(t) * TrainYref.* predref)
        Dhis(:, t) = D
        D = D.*dUpdata
        D = D / sum(D) % updata D       
        TrainClass(:, t) = weight(t) * pred % accumulate prediction of training set
        TestClass(:, t) = weight(t) * round(sigmoid(theta'*TestX'))' % accumulate prediction of test set
        TrainPred = round(sum(TrainClass')' / sum(weight))
        accTrain(t) = sum(TrainPred==TrainY) / nTrain % accuracy of prediction of training set
        TestPred = round(sum(TestClass')' / sum(weight))
        accTest(t) = sum(TestPred==TestY) / nTest % accuracy of prediction of training set
    end
%     TrainPred = round(sum(TrainClass')' / sum(weight))
%     accTrain = sum(TrainPred==TrainY) / nTrain % accuracy of prediction of training set
%     TestPred = round(sum(TestClass')' / sum(weight))
%     accTest = sum(TestPred==TestY) / nTest % accuracy of prediction of training set
    
    output = [accTrain accTest]
    fname = sprintf('result/Acc_%s_%s', method, TrainName)
    save(fname, 'output', '-ascii')
    
    save 'd-out.txt' 'Dhis' -ascii
    save 'we-out.txt' 'weight' -ascii
end