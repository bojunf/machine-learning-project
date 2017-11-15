function [] = Adaboost(TrainName, TestName, T)
    Train = importdata(TrainName) % import training set
    nTrain = size(Train(:, 1), 1) % number of training data
    ndim = size(Train(1, :), 2) - 1 % number of feature
    TrainX = Train(:, 1:ndim) 
    TrainY = Train(:, ndim+1) % separate X and Y
    TrainY = TrainY * 2 - 1
    Test = importdata(TestName) % import test set
    TestX = Test(:, 1:ndim)
    TestY = Test(:, ndim+1)
    TestY = TestY * 2 - 1
    nTest = size(TestX(:, 1), 1) % number of tests data
    D = ones(nTrain, 1) / nTrain % initialize D
%    Dcopy = D
%    Dhist = zeros(nTrain, T)
%    stumphist = zeros(4, T)
%    errhist = zeros(T, 1)
%    bestpredhist = zeros(nTrain, T)

    weight = zeros(T, 1)
    classifier = zeros(T, 4)
    TrainClass = zeros(nTrain, 1)
    TestClass = zeros(nTest, 1)
    accTrain = zeros(T, 1)
    accTest = zeros(T, 1)
    outT = zeros(T, 1)
    for t = 1:T
        [bestStump, minErr, bestPred] = DecisionStump(TrainX, TrainY, D) % prediction based on current D
        weight(t) = 0.5 * log((1 - minErr)/ max(minErr, 1e-16)) % save weight of this stump
        classifier(t, :) = bestStump % save stump of this t
        dUpdata = exp(-weight(t) * TrainY.*bestPred)
        D = D.*dUpdata
        D = D / sum(D) % updata D
        
%        Dhist(:, t) = D
%        stumphist(:, t) = bestStump
%        errhist(t) = minErr
%        bestpredhist(:, t) = bestPred
        
        TrainClass = TrainClass + weight(t) * bestPred % accumulate prediction of training set
        TestClass = TestClass + weight(t) * Classify(TestX, bestStump) % accumulate prediction of test set
        accTrain(t) = sum(sign(TrainClass) == TrainY) / nTrain
        accTest(t) = sum(sign(TestClass) == TestY) / nTest
        outT(t) = t
    end
%     TrainClass = sign(TrainClass)
%     TestClass = sign(TestClass)
%     accTrain = sum(TrainClass==TrainY) / nTrain % accuracy of prediction of training set
%     accTest = sum(TestClass == TestY) / nTest
%    disp(accTrain)
    Testout = [outT accTrain accTest] % output prediction of test set
    save Acc.txt Testout -ascii
    
%    disp(Dhist)
%    disp(stumphist)
%    disp(errhist)
%    disp(weight)
%    disp(bestpredhist)
%    disp(TrainClass)
%    disp(TrainY)
%    disp(TrainX)
%    disp(Dcopy)
end