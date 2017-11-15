function [bestStump, minErr, bestPred] = DecisionStump(X, Y, D)
   ndata = size(X(:, 1), 1) % # of data
   ndim = size(X(1, :), 2) % # of features 
   nstep = 10 % # of step to make the hyperplane
   minErr = 999999999999 % initialize minimum error
   bestStump = zeros(1, 4)
   for i = 1:ndim
      minx = min(X(:, i)) % minimum x value of this dimension
      maxx = max(X(:, i)) % maximum ~~~~
      step = (maxx-minx) / nstep % stepsize
      for j = -1:nstep % make this many try to find minimum error
        for l = 0:1 % left side value loop
            leftval = (l-0.5) * 2 
            rightval = -leftval
            dv = minx + j * step % decision value
            pred = zeros(ndata, 1)
            for k = 1:ndata % prediction of data
               if  X(k, i) < dv
                   pred(k) = leftval
               else
                   pred(k) = rightval
               end
            end
            Err = (pred ~= Y) % error of prediction
            weightErr = D' * Err % weighted error
            if weightErr < minErr % if this data has smaller weighted error, make it minimum point, and save stump and prediction
                minErr = weightErr
                bestPred = pred
                bestStump(1) = i
                bestStump(2) = dv
                bestStump(3) = leftval
                bestStump(4) = rightval
            end
        end
      end
   end
end