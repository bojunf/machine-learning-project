function [pred] = Classify(X, Stump)
    ndata = size(X(:, 1), 1) % number of data
    pred = zeros(ndata, 1)
    for i = 1:ndata % prediction process
       if (X(i, Stump(1)) < Stump(2))
            pred(i) = Stump(3)
       else
            pred(i) = Stump(4) 
       end
    end
end