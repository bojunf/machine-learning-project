function [] = Plot(fname)
    data = importdata(fname) % import data
    X = data(:, 1:2)
    Y = data(:, 3)
    ndata = size(X(:, 1), 1)
    Xplus = []
    Xminus = []
    for i = 1:ndata % separate data into different Y values
%        if i == 1 || i == 3
%            continue
%        end
        if Y(i) == -1
            Xminus = [Xminus; X(i, :)]
        else
            Xplus = [Xplus; X(i, :)]
        end
    end
    scatter(Xplus(:, 1), Xplus(:, 2), 'o')
    hold on
    scatter(Xminus(:, 1), Xminus(:, 2), '*')
end