function [] = KMeans(input, k)

    Train = importdata(input)
    ntrain = length(Train(:, 1))
    nfeat = length(Train(1, :)) - 1
    
    X = Train(:, 1:nfeat)
    Y = Train(:, nfeat+1)
    
    [idX, C] = kmeans(X, k)
    
    save 'out.txt' 'idX' '-ascii'

end