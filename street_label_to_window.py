## @package street_label_to_window
#  Constructs window files

#Self imports
import street_params as sp
import street_labels as sls
import street_utils as su
#Other imports
import pickle
import numpy as np
from os import path as osp
import my_pycaffe_io as mpio
import subprocess
from multiprocessing import Pool
import copy

##
#Make a windown file per folder
def make_window_file_by_folderid(prms, folderId, maxGroups=None):
	if len(prms.labelNames)==1 and prms.labelNames[0] == 'nrml':
		numImPerExample = 1
	else:
		numImPerExample = 2	

	#Assuming the size of images
	h, w, ch = prms.rawImSz, prms.rawImSz, 3
	hCenter, wCenter = int(h/2), int(w/2)
	cr = int(prms.crpSz/2)
	minH = max(0, hCenter - cr)
	maxH = min(h, hCenter + cr)
	minW = max(0, wCenter - cr)
	maxW = min(w, wCenter + cr)  

	#Get the im-label data
	lb, prefix = sls.get_label_by_folderid(prms, folderId, maxGroups=maxGroups)
	#For the imNames
	imNames1 = []
	print('Window file for %s' % folderId)
	imKeys   = pickle.load(open(prms.paths.proc.im.folder.keyFile % folderId, 'r'))
	imKeys   = imKeys['imKeys']
	for pref in prefix:
		tmpNames = []
		_,p1,_,p2 = pref
		tmpNames.append(osp.join(folderId, imKeys[p1]))
		if p2 is not None:
			tmpNames.append(osp.join(folderId, imKeys[p2]))
		imNames1.append(tmpNames) 

	#Randomly permute the data
	N = len(imNames1)
	randState = np.random.RandomState(19)
	perm      = randState.permutation(N) 
	#The output file
	wFile     = prms.paths.exp.window.folderFile % folderId
	wDir,_    = osp.split(wFile)
	sp._mkdir(wDir)
	gen = mpio.GenericWindowWriter(wFile,
					len(imNames1), numImPerExample, prms['labelSz'])
	for i in perm:
		line = []
		for n in range(numImPerExample):
			line.append([imNames1[i][n], [ch, h, w], [minW, minH, maxW, maxH]])
		gen.write(lb[i], *line)
	gen.close()

def _make_window_file_by_folderid(args):
	make_window_file_by_folderid(*args)

##
#Make window files for multiple folders
def make_window_files_geo_folders(prms, isForceWrite=False, maxGroups=None):
	keys   = su.get_geo_folderids(prms)
	print keys
	inArgs = []
	for k in keys:
		if not isForceWrite:
			wFile     = prms.paths.exp.window.folderFile % k
			if osp.exists(wFile):
				print ('Window file for %s exists, skipping rewriting' % wFile)
				continue
		inArgs.append([prms, k, maxGroups])
	pool = Pool(processes=6)
	jobs = pool.map_async(_make_window_file_by_folderid, inArgs)
	res  = jobs.get()
	del pool		

##
#Combine the window files
def make_combined_window_file(prms, setName='train'):
	keys = sp.get_train_test_defs(prms.geoFence, setName=setName)
	wObjs, wNum = [], []
	numIm = None
	for i,k in enumerate(keys):
		wFile  = prms.paths.exp.window.folderFile % k
		wObj   = mpio.GenericWindowReader(wFile)
		wNum.append(wObj.num_)
		wObjs.append(wObj)
		if i==0:
			numIm = wObj.numIm_
		else:
			assert numIm==wObj.numIm_, '%d, %d' % (numIm, wObj.num_)
	
	nExamples  = sum(wNum)
	N = min(nExamples, int(prms.splits.num[setName]))
	mainWFile = mpio.GenericWindowWriter(prms['paths']['windowFile'][setName],
					N, numIm, prms['labelSz'])

	print ('Total examples to chose from: %d' % sum(wNum))	
	wCount = copy.deepcopy(np.array(wNum))
	wNum = np.array(wNum).astype(float)
	wNum = wNum/sum(wNum)
	pCum = np.cumsum(wNum)
	print (pCum)
	assert pCum==1, 'Something is wrong'
	randState = np.random.RandomState(31)
	ignoreCount = 0
	
	nrmlPrune = False
	if 'nrml' in prms.labelNames and len(prms.labelNames)==1:
		if prms.nrmlMakeUni is not None:
			idx = prms.labelNames.index('nrml')
			lbInfo = prms.labels[idx]
			nrmlPrune = True
			if lbInfo.loss_ in ['l2', 'l1']:
				nrmlBins  = np.linspace(-180,180,101)
				binCounts = np.zeros((2,101))
			elif lbInfo.loss_ == 'classify':
				nrmlBins  = np.array(range(lbInfo.numBins_+1))
				binCounts = np.zeros((2,lbInfo.numBins_))
			mxBinCount = int(prms.nrmlMakeUni * np.sum(wCount))
			print ('mxBinCount :%d' % mxBinCount)

	writeCount = 0
	for i in range(N):
		sampleFlag = True
		idx  = None
		while sampleFlag:
			rand = randState.rand()
			idx  = find_first_false(rand >= pCum)
			if not wObjs[idx].is_eof():
				sampleFlag = False
			else:
				ignoreCount += 1
				if ignoreCount > 2000:
					print (ignoreCount, 'Resetting prob distribution')			
					pCum = np.cumsum(wCount/float(sum(wCount)))
					print pCum
					ignoreCount = 0	
	
		wCount[idx] -= 1	
		imNames, lbls = wObjs[idx].read_next()
		if nrmlPrune:
			nrmlIdx   = randState.permutation(2)[0]
			binIdx    = find_bin_index(nrmlBins,lbls[0][nrmlIdx])
			if binCounts[nrmlIdx][binIdx] < mxBinCount:
				binCounts[nrmlIdx][binIdx] += 1
			else:
				continue		
		try:	
			mainWFile.write(lbls[0], *imNames)
		except:
			print 'Error'
			pdb.set_trace()
		writeCount += 1	
	mainWFile.close()
	#Get the count correct for nrmlPrune scenarios
	if nrmlPrune:
		imNames, lbls = [], []
		mainWFile = mpio.GenericWindowReader(prms.paths.windowFile[setName])
		readFlag  = True
		readCount = 0
		while readFlag:
			name, lb = mainWFile.read_next()
			imNames.append(name)
			lbls.append(lb)
			readCount += 1
			if readCount == writeCount:
				readFlag = False
		mainWFile.close()
		#Write the corrected version
		mainWFile = mpio.GenericWindowWriter(prms['paths']['windowFile'][setName],
						writeCount, numIm, prms['labelSz'])
		for n in range(writeCount):
			mainWFile.write(lbls[n][0], *imNames[n])
		mainWFile.close()

##
#Process the normals prototxt
def get_binned_normals_from_window_file(prms, setName='test'):
	wFileName = prms.paths.windowFile[setName]
	wFid      = mpio.GenericWindowReader(wFileName)
	lbls      = wFid.get_all_labels() 
	N, nLb    = lbls.shape
	nLb -= 1
	nBins  = 100
	binned = []
	for n in range(nLb):
		binned.append(np.histogram(lbls[:,n], 100))
	return binned
		 
##
#Get the statistics of labels
def get_label_stats(prms, isForceCompute):
	#Get foldernames in the training et

	#Find labels

	#Sample 10% of the labels - compute mean and var

	#Store the label info	
