function [] = LoopLogistic(Trainfile, Testfile, LoopObj, LoopArray, otherValue1, otherValue2, method)
    
    Acc = zeros(length(LoopArray), 1)
    AccTest = zeros(length(LoopArray), 1)
    
    if strcmp(LoopObj, 'alpha')
        for i = 1:length(LoopArray)
            alpha = LoopArray(i)
            [Acc(i), AccTest(i)] = Logistic(Trainfile, Testfile, alpha, otherValue1, method, otherValue2)
        end
        output = [LoopArray Acc AccTest]
        fname = sprintf('Acc_alpha_%s.txt', method)
        save(fname, 'output', '-ascii')
    elseif strcmp(LoopObj, 'niter')
        for i = 1:length(LoopArray)
            niter = LoopArray(i)
            [Acc(i), AccTest(i)] = Logistic(Trainfile, Testfile, otherValue1, niter, method, otherValue2)
        end
        output = [LoopArray Acc AccTest]
        fname = sprintf('Acc_niter_%s.txt', method)
        save(fname, 'output', '-ascii')
    elseif strcmp(LoopObj, 'lambda')
        for i = 1:length(LoopArray)
            lambda = LoopArray(i)
            [Acc(i), AccTest(i)] = Logistic(Trainfile, Testfile, otherValue1, otherValue2, method, lambda)
        end
        output = [LoopArray Acc AccTest]
        fname = sprintf('result-borm/Acc_lamda_%s.txt', method)
        save(fname, 'output', '-ascii')
    else
        disp('Wrong loop object')
        return
    end
end