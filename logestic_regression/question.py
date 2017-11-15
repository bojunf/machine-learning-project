import sys

fname = str(sys.argv[1])

tmpq, tmpn = [], []

with open('{0}'.format(fname), 'r') as f:
	for line in f.readlines():
		arr = line.replace('\n', '').split(',')
		FlagQ = False
		for i in range(len(arr)):
			if (arr[i] == '?'):
				FlagQ = True
				break
		if (FlagQ):
			tmpq.append(line)
		else:
			tmpn.append(line)

file('data-sure.txt', 'w').writelines(tmpn)
file('data-question.txt', 'w').writelines(tmpq)