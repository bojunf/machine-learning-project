function [] = LoopLogistic(Trainfile, LoopObj, LoopArray, otherValue, method)
    
    Acc = zeros(length(LoopArray), 1)
    
    if strcmp(LoopObj, 'alpha')
        for i = 1:length(LoopArray)
            alpha = LoopArray(i)
            Acc(i) = Logistic(Trainfile, alpha, otherValue, method)
        end
        output = [LoopArray Acc]
        fname = sprintf('Acc_alpha_%s.txt', method)
        save(fname, 'output', '-ascii')
    elseif strcmp(LoopObj, 'niter')
        for i = 1:length(LoopArray)
            niter = LoopArray(i)
            Acc(i) = Logistic(Trainfile, otherValue, niter, method)
        end
        output = [LoopArray Acc]
        fname = sprintf('Acc_niter_%s.txt', method)
        save(fname, 'output', '-ascii')
    else
        disp('Wrong loop object')
        return
    end
end