function [] = LoopNhAndL(hidden1, hidden2, alpha, Trainfile, Testfile)

nhidden1 = length(hidden1)
nhidden2 = length(hidden2)
nalpha = length(alpha)

for i = 1:nhidden1 % loop over all combinations of number of hidden units and alpha values
    for k = 1:nhidden2
        for j = 1:nalpha
%        disp(hidden(i))
            NNsigCl(hidden1(i), hidden2(k), alpha(j), Trainfile, Testfile)
        end
    end
end


end