import numpy as np
import matplotlib.pyplot as plt

def get_experiment_accuracy(exp, lossName=None):
	logFile = exp.expFile_.logTrain_	
	lossNames = lossName
	return log2loss(logFile, lossNames)

def plot_experiment_accuracy(exp, svFile=None,
								isTrainOnly=False, isTestOnly=False, ax=None,
								lossName=None):
	testData, trainData = get_experiment_accuracy(exp, lossName=lossName)
	if ax is None:
		plt.figure()
		ax = plt.subplot(111)
	if not isTrainOnly:
		for k in testData.keys():
			if lossName is not None and not (k in lossName):
				continue
			if k == 'iters':
				continue
			ax.plot(testData['iters'][1:], testData[k][1:],'r',  linewidth=3.0)
	if not isTestOnly:
		for k in trainData.keys():
			if lossName is not None and not (k in lossName):
				continue
			if k == 'iters':
				continue
			ax.plot(trainData['iters'][1:], trainData[k][1:],'b',  linewidth=3.0)
	if svFile is not None:
		with PdfPages(svFile) as pdf:
			pdf.savefig()
	return ax


def read_log(fileName):
	'''
	'''
	fid = open(fileName,'r')
	trainLines, trainIter = [], []
	testLines, testIter   = [], []
	iterNum   = None
	iterStart = False
	#Read the test lines in the log
	while True:
		try:
			l = fid.readline()
			if not l:
				break
			if 'Iteration' in l:
				iterNum  = int(l.split()[5][0:-1])
				iterStart = True
			if 'Test' in l and ('loss' in l or 'acc' in l):
				testLines.append(l)
				if iterStart:
					testIter.append(iterNum)
					iterStart = False
			if 'Train' in l and ('loss' in l or 'acc' in l):
				trainLines.append(l)
				if iterStart:
					trainIter.append(iterNum)
					iterStart = False
		except ValueError:
			print 'Error in reading .. Skipping line %s ' % l
	fid.close()
	return testLines, testIter, trainLines, trainIter

##
#Read the loss values from a log file
def log2loss(fName, lossNames):
	testLines, testIter, trainLines, trainIter = read_log(fName)
	N = len(lossNames)
	#print N, len(testLines), testIter
	#assert(len(testLines)==N*len(testIter), 'Error in test Lines')
	#assert(len(trainLines)==N*len(trainIter), 'Error in train lines')
		
	testData, trainData = {}, {}
	for t in lossNames:
		testData[t], trainData[t] = [], []
		#Parse the test data
		for l in testLines:
			if t in l:
				data = l.split()
				#print data
				assert data[8] == t, data
				idx = data.index('=')
				testData[t].append(float(data[idx+1]))
		#Parse the train data
		for l in trainLines:
			if t in l:
				data = l.split()
				assert data[8] == t
				idx = data.index('=')
				trainData[t].append(float(data[idx+1]))
	for t in lossNames:
		testData[t]  = np.array(testData[t])
		trainData[t] = np.array(trainData[t])
	testData['iters']  = np.array(testIter)
	trainData['iters'] = np.array(trainIter)
	return testData, trainData

