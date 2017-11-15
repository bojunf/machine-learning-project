import sys
import numpy as np

ftrain = str(sys.argv[1])
ftest = str(sys.argv[2])
#fval = str(sys.argv[3])  # input file names


traindata = []
with open('{0}'.format(ftrain), 'r') as f: # read training data
	nline = 0
	for line in f.readlines():
		nline = nline + 1
		arr = line.replace('\n', '').split(' ')
		traindata.append(map(float, arr))

traindata = np.array(traindata)

mean, std = [], []


nfeat = len(traindata[0]) - 1

for i in range(nfeat): # find mean and std for each features of all training data
	mean.append(np.mean(traindata[:, i]))
	std.append(np.std(traindata[:, i]))

testdata, valdata = [], []

normtrain, normtest, normval = [], [], []

with open('{0}'.format(ftest), 'r') as f: # read test data
	nline = 0
	for line in f.readlines():
		nline = nline + 1
		arr = line.replace('\n', '').split(' ')
		testdata.append(map(float, arr))

# with open('{0}'.format(fval), 'r') as f: # read validation data
# 	nline = 0
# 	for line in f.readlines():
# 		nline = nline + 1
# 		arr = line.replace('\n', '').split(',')
# 		valdata.append(map(int, arr))

testdata = np.array(testdata)
#valdata = np.array(valdata)


for i in range(nfeat): # normalize data based on mean and std of training data
	if (std[i] != 0.0):
		traindata[:, i] = (traindata[:, i] - mean[i]) / float(std[i])
		testdata[:, i] = (testdata[:, i] - mean[i]) / float(std[i])
#		valdata[:, i] = (valdata[:, i] - mean[i]) / float(std[i])


np.savetxt('norm-wpbc-train.txt', traindata)
np.savetxt('norm-wpbc-test.txt', testdata)
#np.savetxt('norm_val.txt', valdata)

#np.savetxt('mean.txt', mean)
#np.savetxt('std.txt', std) # save normalized data into files




