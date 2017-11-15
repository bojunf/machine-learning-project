function [] = LoopNhAndL(hidden, alpha, Trainfile, Testfile)

nhidden = length(hidden)
nalpha = length(alpha)

for i = 1:nhidden % loop over all combinations of number of hidden units and alpha values
    for j = 1:nalpha
%        disp(hidden(i))
        NNsigCl(hidden(i), alpha(j), Trainfile, Testfile)
    end
end


end