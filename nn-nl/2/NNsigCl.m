function [] = NNsigCl(hidden_size, alpha, Trainfile, Testfile)        Train = importdata(Trainfile)    ntrain = length(Train(:, 1))    nfeat = length(Train(1, :)) - 1    Xtrain = Train(:, 1:nfeat)    Ytrain = Train(:, nfeat+1)        Test = importdata(Testfile)    ntest = length(Test(:, 1))    Xtest = Test(:, 1:nfeat)    Ytest = Test(:, nfeat+1)        Xtrain = [ones(ntrain, 1) Xtrain]    Xtest = [ones(ntest, 1) Xtest] % add bias term            lambda = 0.0    niter = 200            [thetaih, thetaho] = InitWeight(nfeat, hidden_size, 2) % initialize weights            [L_history, thetaho, thetaih] = nnGD(Xtrain, Ytrain, thetaih, thetaho, lambda, niter, alpha) % use gradient descent to get optimized weights and loss function            probTrain = sigmoid([ones(ntrain, 1) sigmoid(Xtrain * thetaih)] * thetaho)        predTrain = (probTrain(:, 2) > probTrain(:, 1))        probTest = sigmoid([ones(ntest, 1) sigmoid(Xtest * thetaih)] * thetaho)        predTest = (probTest(:, 2) > probTest(:, 1))    %     probVal = sigmoid([ones(nval, 1) sigmoid(Xval * thetaih)] * thetaho)%     %     predVal= (probVal(:, 2) > probVal(:, 1)) % make prediction for three data sets based on weights by gradient descent        Acc_Train = sum(predTrain == Ytrain) / ntrain        Acc_Test = sum(predTest == Ytest) / ntest    %    Acc_Val = sum(predVal == Yval) / nval % calculate accuracy of prediction of each data sets        output = [Acc_Train; Acc_Test]       fname1 = sprintf('result-wpbc/L_train_%d_%.3f.txt', hidden_size, alpha)%    fname2 = sprintf('L_val_%d_%.2f.txt', hidden_size, alpha)    fname3 = sprintf('result-wpbc/Acc_%d_%.3f.txt', hidden_size, alpha)    save(fname1, 'L_history', '-ascii')%    save(fname2, 'LVal_history', '-ascii')    save(fname3, 'output', '-ascii') % output data neededend