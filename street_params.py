## @package street_params
#	Parameter settings

import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess
import matplotlib.pyplot as plt
import mydisplay as mydisp
#import h5py as h5
import pickle
import matplotlib.path as mplPath
import re
import street_utils as su

def _mkdir(fName):
	if not osp.exists(fName):
		os.makedirs(fName)

##
#Read the coordinates of DC that need to be geofenced. 
def read_geo_coordinates(fName):
	coords   = []
	readFlag = False
	with open(fName,'r') as f:
		lines = f.readlines()
		for l in lines:
			#Detect the end of a set of coordinates
			if 'coordinates' in l	and readFlag==True:
				readFlag = False
				continue
			if readFlag:
				#print l	
				coords.append([float(c) for c in re.split('\,+|\ +', l.strip())])
			if 'coordinates' in l and not readFlag:
				readFlag = True
	cArr = []
	for c in coords:
		cArr.append(mplPath.Path(np.array(c).reshape((len(c)/3,3))[:,0:2]))
	return cArr


def get_paths():
	paths      = edict()
	#For storing the directories
	paths.dirs = edict()
	#The raw data
	paths.dataDr  = osp.join(cfg.STREETVIEW_DATA_MAIN, 'pulkitag/data_sets/streetview')
	paths.raw     = edict()
	paths.raw.dr  = osp.join(paths.dataDr, 'raw')

	#For reading data faster we will use a different data direcoty
	imReadDr = osp.join(cfg.STREETVIEW_DATA_READ_IM, 'pulkitag/data_sets/streetview')

	#Processed data
	paths.proc    = edict()
	paths.proc.dr = osp.join(paths.dataDr, 'proc')
	#Raw Tar Files
	paths.tar     = edict()
	paths.tar.dr  = osp.join(paths.dataDr, 'tar')
	#The Code directory
	paths.code    = edict()
	paths.code.dr = cfg.STREETVIEW_CODE_PATH
	paths.baseNetsDr = osp.join(paths.code.dr, 'base_files')
	#List of tar files
	paths.tar.fileList = osp.join(paths.code.dr, 'data_list.txt') 

	#Store the names of all the folders
	paths.proc.folders    = edict()
	paths.proc.folders.dr = osp.join(paths.proc.dr, 'folders')
	_mkdir(paths.proc.folders.dr)
	#Stores the folder names along with the keys
	paths.proc.folders.key  = osp.join(paths.proc.folders.dr, 'key.txt') 
	paths.proc.folders.pre  = osp.join(paths.proc.folders.dr, '%s.txt') 
	#The keys for the algined folders
	paths.proc.folders.aKey  = osp.join(paths.proc.folders.dr, 'key-aligned.txt') 
	#Keys for non-algined folders
	paths.proc.folders.naKey = osp.join(paths.proc.folders.dr, 'key-non_aligned.txt') 

	#Storing resized images
	imProcDr         = osp.join(imReadDr, 'proc')
	_mkdir(imProcDr)
	paths.proc.im    =  edict()
	paths.proc.im.dr =  osp.join(imProcDr, 'resize-im')
	_mkdir(paths.proc.im.dr)
	paths.proc.im.keyFile = osp.join(imProcDr, 'im%d-keys.pkl') 
	paths.proc.im.dr      = osp.join(paths.proc.im.dr, 'im%d')
	#Count the number of keys already stored - useful for appending the files.
	#Note that this count maynot be accurate but will be larger than the total number
	#of images saved 	
	paths.proc.im.keyCountFile = osp.join(paths.proc.im.dr, 'im%d-key-count.pkl') 
	#Storing images folderwise
	paths.proc.im.folder = edict()
	folderDr = osp.join(paths.proc.im.dr, '%s')
	paths.proc.im.folder.keyFile = osp.join(folderDr, 'keys.pkl')
	paths.proc.im.folder.dr      = folderDr
	paths.proc.im.folder.tarFile = osp.join(paths.proc.im.dr, '%s.tar')
			
	#Count info
	paths.proc.countFile = osp.join(paths.proc.folders.dr, 'counts.h5')	

	#Get the file containing the split data
	splitsDr = osp.join(paths.proc.dr, 'train_test_splits')
	_mkdir(splitsDr)
	paths.proc.splitsFile = osp.join(splitsDr, '%s.pkl') 

	#Label data
	paths.label    = edict()
	paths.label.dr   = osp.join(paths.proc.dr, 'labels')
	#Store the normals
	nrmlDir          = osp.join(paths.label.dr, 'nrml')
	_mkdir(nrmlDir)
	paths.label.nrml = osp.join(nrmlDir, '%s.txt')
	#The data is chunked as groups - so we will save them 
	grpsDir          = osp.join(paths.label.dr, 'groups')
	_mkdir(grpsDir)
	paths.label.grps = osp.join(grpsDir, '%s.pkl')
	#Save the groups containing alignment data only
	grpsAlgnDir      = osp.join(paths.label.dr, 'groups-aligned')
	_mkdir(grpsAlgnDir)
	paths.label.grpsAlgn = osp.join(grpsAlgnDir, '%s.pkl')
		
	paths.exp    = edict()
	paths.exp.dr = osp.join(paths.dataDr, 'exp')
	_mkdir(paths.exp.dr)
	#Window data file
	paths.exp.window    = edict()
	paths.exp.window.dr = osp.join(paths.exp.dr, 'window-files')
	_mkdir(paths.exp.window.dr) 
	paths.exp.window.tr = osp.join(paths.exp.window.dr, 'train-%s.txt')
	paths.exp.window.te = osp.join(paths.exp.window.dr, 'test-%s.txt')
	#Folderwise window dir
	paths.exp.window.folderDr   = osp.join(paths.exp.dr, 'folder-window-files')
	_mkdir(paths.exp.window.folderDr)
	# %s, %s -- folderId, windowFile-str   
	paths.exp.window.folderFile = osp.join(paths.exp.window.folderDr, '%s', '%s.txt')
	
	#Snapshot dir
	paths.exp.snapshot    = edict()
	paths.exp.snapshot.dr = osp.join(paths.exp.dr, 'snapshots') 
	_mkdir(paths.exp.snapshot.dr)

	#Paths for geofence data
	paths.geoFile = osp.join(paths.code.dr, 'geofence', '%s.txt')

	#Storing the results.
	paths.res    = edict()
	paths.res.dr = osp.join(paths.dataDr, 'res')
	_mkdir(paths.res.dr)
	paths.resFile = osp.join(paths.res.dr, '%s.pkl')
	paths.res.testImVisDr = osp.join(paths.res.dr, 'test-imvis')
	_mkdir(paths.res.testImVisDr)
	#paths.res.testImVis   = osp.join(paths.res.testImVisDr, 'im%05d.jpg')   

	paths.grp = edict()
	paths.grp.keyStr    = '%07d'
	paths.grp.geoDr     = osp.join(paths.res.dr, 'geo-grp') 
	_mkdir(paths.grp.geoDr)
	paths.grp.geoFile   = osp.join(paths.grp.geoDr, '%s', '%s.pkl')

	#For legacy reasons
	paths.expDir  = osp.join(paths.exp.dr, 'caffe-files')
	paths.snapDir = paths.exp.snapshot.dr
	return paths

##
# Get the label dimensions
def get_label_size(labelClass, labelType):
	if labelClass == 'nrml':
		if labelType == 'xyz':
			#The third is always 0 as it is defined by the axis of the camera
			#that points at the target. 
			lSz = 2
		else:
			raise Exception('%s,%s not recognized' % (labelClass, labelType))
	elif labelClass == 'ptch':
		if labelType in ['wngtv', 'hngtv']:
			#It is just going to be a softmax loss
			lSz = 1
		else:
			raise Exception('%s,%s not recognized' % (labelClass, labelType))
	elif labelClass == 'pose':
		if labelType in ['quat']:
			lSz = 4
		elif labelType in ['euler']:
			#One of the angles is always 0
			lSz = 2
		elif labelType in ['euler-5dof']:
			lSz = 5
		elif labelType in ['euler-6dof']:
			lSz = 6
		else:
			raise Exception('%s,%s not recognized' % (labelClass, labelType))
	else:
		raise Exception('%s not recognized' % labelClass)
	return lSz

##
class LabelNLoss(object):
	def __init__(self, labelClass, labelType, loss, 
							ptchPosFrac=0.5, maxRot=None, numBins=20, 
							binType=None, isMultiLabel=False):
		'''
			ptchPosFrac: When considering patch matching data - the fraction of patches
									 to consider as positives
			maxRot:      When considering the pose - the maximum rotation in degrees 
									 between the image pairs
									 that need to be considered. 
		'''
		self.label_     = labelClass
		self.labelType_ = labelType
		self.loss_      = loss
		self.isMultiLabel = isMultiLabel
		self.numBins_ = numBins
		self.binType_ = binType
		assert self.loss_ in ['l2', 'classify', 'l1'], self.loss_
		#augLbSz_ - augmented labelSz to include the ignore label option
		self.augLbSz_, self.lbSz_  = self.get_label_sz()
		self.lbStr_       = '%s-%s' % (self.label_, self.labelType_)
		#Patch Labels
		if labelClass == 'ptch':
			self.posFrac_ = ptchPosFrac
			self.lbStr_   = self.lbStr_ + '-posFrac%.1f' % self.posFrac_ 	
	
		#Pose Labels
		if labelClass == 'pose':
			self.maxRot_  = maxRot 
			if maxRot is not None:
				self.lbStr_   = self.lbStr_ + '-mxRot%d' % maxRot
			if self.loss_ in ['classify']:
				self.lbStr_   = self.lbStr_ + 'classify'
			if self.binType_ is not None:
				self.lbStr_ = self.lbStr_ + '-%s-nBins-%d' % (binType, numBins)
				if self.labelType_ == 'quat':
					self.binRange_ = np.linspace(-1, 1, self.numBins_+1)	
				elif self.labelType_ == 'euler':
					if maxRot is None:
						self.binRange_ = np.linspace(-180, 180, self.numBins_+1)
					else:
						self.binRange_ = np.linspace(-maxRot, maxRot, self.numBins_+1)
		#Nrml Labels
		if labelClass == 'nrml':
			if self.binType_ is not None:
				self.lbStr_ = self.lbStr_ + '%s-nBins-%d' % (binType, numBins)
				self.binRange_ = np.linspace(-1, 1, self.numBins_+1)	
	
	def get_label_sz(self):
		lbSz = get_label_size(self.label_, self.labelType_)
		if self.binType_ is None:
			if self.loss_ == 'classify':
				augLbSz = lbSz
			else:
				augLbSz  = lbSz + 1
		else:
			augLbSz, lbSz = lbSz, lbSz
		return augLbSz, lbSz


def get_train_test_defs(geoFence, ver='v1', setName=None):
	if geoFence == 'dc-v1':
		if ver=='v1':
			trainFolderKeys = ['0048']
			testFolderKeys  = ['0052'] 
		else:
			raise Exception('%s not recognized' % v1)
	elif geoFence == 'dc-v2':
		geoFile = 'geofence/dc-v2.txt'
		keys = []
		with open(geoFile,'r') as fid:
			lines = fid.readlines()
			for l in lines:
				key, _ = l.strip().split()
				keys.append(key)	
			testFolderKeys = ['0008']
			trainFolderKeys = [k for k in keys if k not in testFolderKeys]
	elif geoFence == 'cities-v1':
		geoFile = 'geofence/cities-v1.txt'
		keys = []
		with open(geoFile,'r') as fid:
			lines = fid.readlines()
			for l in lines:
				key, _ = l.strip().split()
				keys.append(key)	
			testFolderKeys = ['0070']
			ignoreKeys     = ['0008'] + testFolderKeys
			trainFolderKeys = [k for k in keys if k not in ignoreKeys]
	elif geoFence == 'vegas-v1':
		geoFile = 'geofence/vegas-v1.txt'
		keys = []
		with open(geoFile,'r') as fid:
			lines = fid.readlines()
			for l in lines:
				key, _ = l.strip().split()
				keys.append(key)	
			testFolderKeys = keys
			trainFolderKeys = []
	else:
		raise Exception('%s not recognized' % geoFence)
	if setName == 'train':
		return trainFolderKeys
	elif setName == 'test':
		return testFolderKeys
	elif setName is None:
		return trainFolderKeys, testFolderKeys

##
#get prms
def get_prms(isAligned=True, 
						 labels=['nrml'], labelType=['xyz'], 
						 lossType=['l2'], labelFrac=[1],
						 nBins=[20], binTypes=[None], 
						 labelNrmlz=None, 
						 crpSz=101,
						 numTrain=1e+06, numTest=1e+04,
						 trnSeq=[], 
						 tePct=1.0, teGap=5,
						 ptchPosFrac=0.5, maxEulerRot=None, 
						 geoFence='dc-v1', rawImSz=640,
						 splitDist=None, splitVer='v1',
						 nrmlMakeUni=0.002, mxPtchRot=None,
						 isV2=False):
	'''
		labels    : What labels to use - make it a list for multiple
								kind of labels
								 nrml - surface normals
									 xyz - as nx, ny, nz
								 ptch: patch matching
									 wngtv: weak negatives 
									 hngtv: hard negatices
								 pose - relative pose	
									 euler - as euler angles
									 quat  - as quaternions
		labelNrmlz : Normalization of the labels
								 	None - no normalization of labels
		lossType   : What loss is being used
								 	l2
								 	l1
								 	l2-tukey : l2 loss with tukey biweight
								 	cntrstv  : contrastive
		cropSz      : Size of the image crop to be used.
		tePct       : % of groups to be labelled as test
		teGap       : The number of groups that should be skipped before and after test
									to ensure there are very low chances of overlap
		splitDist   : The minimum distance between the groups in train and test splits
									in meters 
									if splitDist is specified teGap is overriden
		ptchPosFrac : The fraction of the positive patches in the matching
		nrmlMakeUni :	nrmls are highly skewed - so this tries to make the normals uniformly
									distributed.
		isV2        : The version from Feb16 which is free of errors in rotation computation 

		NOTES
		I have tried to form prms so that they have enough information to specify
		the data formation and generation of window files. 
		randomCrop, concatLayer are properties of the training
                            they should not be in prms, but in caffePrms
	'''
	assert type(labels) == list, 'labels must be a list'
	assert type(lossType) == list, 'lossType should be list'
	assert len(lossType) == len(labels)
	assert len(labels)   == len(labelType)
	assert len(labels)  == len(labelFrac)
	assert len(labels)  == len(nBins)
	assert len(labels)  == len(binTypes)
	assert sum(labelFrac)==1, 'Set labelFrac appropriately'
	#Assert that labels are sorted
	sortLabels = sorted(labels)
	for i,l in enumerate(labels):
		assert(l == sortLabels[i])

	paths = get_paths()
	prms  = edict()
	prms.isAligned = isAligned
	prms.rawImSz      = rawImSz
	
	#Label infoo
	prms.labelSz   = 0
	prms.labels    = []
	prms.labelNames, prms.labelNameStr = labels, ''
	prms.labelFrac = labelFrac
	if len(labels) > 1:
		isMultiLabel = True
	else:
		isMultiLabel = False
	prms.isMultiLabel = isMultiLabel
	prms.geoFence     = geoFence
	#Form the label types
	prms.labelSzList = [0]
	isNrml = False
	for lb,lbT,ls,lbF,nbn,bt in zip(labels, labelType, lossType, labelFrac, nBins, binTypes):
		prms.labels = prms.labels + [LabelNLoss(lb, lbT, ls,
										 ptchPosFrac=ptchPosFrac, maxRot=maxEulerRot,
										 isMultiLabel=isMultiLabel, 
										 numBins=nbn, binType=bt	)]
		prms.labelNameStr = prms.labelNameStr + '_%s' % lb
		lbSz              = prms.labels[-1].get_label_sz()[0]
		prms.labelSzList.append(prms.labelSzList[-1] + lbSz)
		prms.labelSz      = prms.labelSz + lbSz
		if lb=='nrml':
			isNrml = True
	prms.labelNameStr = prms.labelNameStr[1:]

	if 'ptch' in labels or 'pose' in labels:
		prms.isSiamese = True
	else:
		prms.isSiamese = False
	prms['lbNrmlz'] = labelNrmlz
	prms['crpSz']   = crpSz
	prms['trnSeq']  = trnSeq
	prms.geoPoly    = None 	
	prms.isV2 = isV2

	prms.splits = edict()
	if splitDist is not None:
		teGap = None
	prms.splits.num = edict()
	prms.splits.num.train = numTrain
	prms.splits.num.test  = numTest
	prms.splits.tePct    = tePct
	prms.splits.teGap    = teGap
	prms.splits.dist     = splitDist
	prms.splits.randSeed = 3
	prms.splits.ver      = splitVer

	#Form the splits file
	if prms.splits.dist is not None:
		if prms.geoFence is None:
			splitsStr = 'spDist%d_spVer-%s'%(prms.splits.dist, prms.splits.ver) 
		else:	
			splitsStr = 'spDist%d_spVer-%s_geo%s'\
									 %(prms.splits.dist, prms.splits.ver, prms.geoFence) 
	else:
		splitsStr = 'tePct%.1f_teGap%d_teSeed%d' % (tePct, teGap, prms.splits.randSeed) 
	paths.proc.splitsFile = paths.proc.splitsFile % (splitsStr + '/%s') 	
	splitDr, _ = osp.split(paths.proc.splitsFile)
	_mkdir(splitDr)

	if prms.isMultiLabel:
		expStr    = ''.join(['%s-frac%.2f_' % (lb.lbStr_, lbf)\
										 for lb,lbf in zip(prms.labels, labelFrac)])
	else:
		expStr    = ''.join(['%s_' % lb.lbStr_ for lb in prms.labels])
	expStr    = expStr[0:-1]
	if prms.splits.dist is not None:
		expStr = '%s_%s' % (expStr,splitsStr)

	if geoFence is not None:
		expStr     = '%s_geo-%s' % (expStr, geoFence)
		paths.geoFile = paths.geoFile % geoFence
		prms.geoPoly  = read_geo_coordinates(paths.geoFile)
	if not(rawImSz==640):
		imStr = '_rawImSz%d' % rawImSz
	else:
		imStr = ''
	prms.mxPtchRot = mxPtchRot
	if mxPtchRot is not None:
		expStr = expStr + ('_mxPtchRot-%d' % mxPtchRot) 

	expName   = '%s_crpSz%d_nTr-%.2e%s' % (expStr, crpSz, numTrain, imStr)
	teExpName = '%s_crpSz%d_nTe-%.2e%s' % (expStr, crpSz, numTest, imStr)
	expName2  = '%s_crpSz%d%s' % (expStr, crpSz, imStr)
	if prms.isV2:
		expName  = '%s_%s' % (expName, 'exp-V2')
		teExpName  = '%s_%s' % (teExpName, 'exp-V2')
		expName2 = '%s_%s' % (expName2, 'exp-V2')
	prms['expName'] = expName

	#Form the window files
	paths['windowFile'] = {}
	windowDir = paths.exp.window.dr
	prms.nrmlMakeUni = nrmlMakeUni
	if isNrml and nrmlMakeUni is not None:
		expName   = expName   + '_nrml-unimax-%.3f' % nrmlMakeUni
		teExpName = teExpName + '_nrml-unimax-%.3f' % nrmlMakeUni  
	paths['windowFile']['train'] = osp.join(windowDir, 'train_%s.txt' % expName)
	paths['windowFile']['test']  = osp.join(windowDir, 'test_%s.txt'  % teExpName)
	paths.exp.window.folderFile  = paths.exp.window.folderFile  %  ('%s', expName2)

	#Files for saving the geolocalized groups
	if prms.geoPoly is not None:
		paths.grp.geoFile = paths.grp.geoFile % (geoFence, '%s')
		geoDirName, _ = osp.split(paths.grp.geoFile)
		_mkdir(geoDirName)

	#Files for storing the resized images
	paths.proc.im.dr       = paths.proc.im.dr % rawImSz
	paths.proc.im.keyFile  = paths.proc.im.keyFile % rawImSz
	paths.proc.im.folder.dr       = paths.proc.im.folder.dr % (rawImSz, '%s')
	paths.proc.im.folder.tarFile  = paths.proc.im.folder.tarFile % (rawImSz, '%s')
	paths.proc.im.folder.keyFile  = paths.proc.im.folder.keyFile % (rawImSz, '%s')

	prms['paths'] = paths
	#Get the pose stats
	prms['poseStats'] = {}
	#prms['poseStats']['mu'], prms['poseStats']['sd'], prms['poseStats']['scale'] =\
	#					get_pose_stats(prms)
	ltStr = ''
	ltFlag = False
	for lt in lossType:
		if lt == 'l1':
			ltFlag = True
			ltStr = 'loss-l1'
	if ltFlag:
		prms['expName'] = '%s_%s' % (prms['expName'], ltStr)
	return prms

##
#Get normals for 
def get_prms_nrml(**kwargs):
	return get_prms(labels=['nrml'], labelType=['xyz'], lossType=['l2'], **kwargs)
		
def get_prms_ptch(**kwargs):
	return get_prms(labels=['ptch'], labelType=['wngtv'], lossType=['classify'], **kwargs)

def get_prms_pose(**kwargs):
	return get_prms(labels=['pose'], labelType=['quat'], lossType=['l2'], **kwargs)

def get_prms_pose_euler(**kwargs):
	return get_prms(labels=['pose'], labelType=['euler'], lossType=['l2'], **kwargs)

def get_prms_vegas_ptch():
	prms = get_prms(labels=['ptch'], labelType=['wngtv'], rawImSz=256, numTest=100000, isAligned=False,
									splitDist=100, geoFence='vegas-v1', crpSz=192)
	return prms
