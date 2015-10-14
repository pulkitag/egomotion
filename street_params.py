import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess
import matplotlib.pyplot as plt
import mydisplay as mydisp
import h5py as h5
import pickle
import matplotlib.path as mplPath
import re

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
	paths.proc.folders.aKey = osp.join(paths.proc.folders.dr, 'key-aligned.txt') 

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

	paths.exp    = edict()
	paths.exp.dr = osp.join(paths.dataDr, 'exp')
	_mkdir(paths.exp.dr)
	#Window data file
	paths.exp.window    = edict()
	paths.exp.window.dr = osp.join(paths.exp.dr, 'window-files')
	_mkdir(paths.exp.window.dr) 
	paths.exp.window.tr = osp.join(paths.exp.window.dr, 'train-%s.txt')
	paths.exp.window.te = osp.join(paths.exp.window.dr, 'test-%s.txt')
	#Snapshot dir
	paths.exp.snapshot    = edict()
	paths.exp.snapshot.dr = osp.join(paths.exp.dr, 'snapshots') 
	_mkdir(paths.exp.snapshot.dr)

	#Paths for geofence data
	paths.geoFile = osp.join(paths.code.dr, 'geofence', '%s.txt')

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
		else:
			raise Exception('%s,%s not recognized' % (labelClass, labelType))
	else:
		raise Exception('%s not recognized' % labelClass)
	return lSz

##
class LabelNLoss(object):
	def __init__(self, labelClass, labelType, loss, ptchPosFrac=0.5, maxRot=None):
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
		#augLbSz_ - augmented labelSz to include the ignore label option
		self.augLbSz_, self.lbSz_  = self.get_label_sz()
		self.lbStr_     = '%s-%s' % (self.label_, self.labelType_)
		if labelClass == 'ptch':
			self.posFrac_ = ptchPosFrac
			self.lbStr_   = self.lbStr_ + '-posFrac%.1f' % self.posFrac_ 	
		if labelClass == 'pose':
			self.maxRot_  = maxRot 
			if maxRot is not None:
				self.lbStr_   = self.lbStr_ + '-mxRot%d' % maxRot
	
	def get_label_sz(self):
		lbSz = get_label_size(self.label_, self.labelType_) 
		if not(self.label_ == 'nrml') and self.loss_ in ['l2', 'l1', 'l2-tukey']:
			#augLbSz = lbSz + 1
			augLbSz  = lbSz
		else:
			augLbSz = lbSz
		return augLbSz, lbSz

##
#get prms
def get_prms(isAligned=True, 
						 labels=['nrml'], labelType=['xyz'], 
						 lossType=['l2'],
						 labelNrmlz=None, 
						 crpSz=101,
						 numTrain=1e+06, numTest=1e+04,
						 trnSeq=[], 
						 tePct=1.0, teGap=5,
						 ptchPosFrac=0.5, maxEulerRot=None, 
						 geoFence=None):
	'''
		labels    : What labels to use - make it a list for multiple
								kind of labels
								 nrml - surface normals
									 xyz - as nx, ny, nz
								 ptch - patch matching
									 wngtv - weak negatives 
									 hngtv = hard negatices
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
		ptchPosFrac : The fraction of the positive patches in the matching

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
	#Assert that labels are sorted
	sortLabels = sorted(labels)
	for i,l in enumerate(labels):
		assert(l == sortLabels[i])

	paths = get_paths()
	prms  = edict()
	prms.isAligned = isAligned
	
	#Label infoo
	prms.labelSz = 0
	prms.labels  = []
	prms.labelNames, prms.labelNameStr = labels, ''
	for lb,lbT,ls in zip(labels, labelType, lossType):
		prms.labels = prms.labels + [LabelNLoss(lb, lbT, ls,
										 ptchPosFrac=ptchPosFrac, maxRot=maxEulerRot)]
		prms.labelNameStr = prms.labelNameStr + '_%s' % lb
		prms.labelSz      = prms.labelSz + prms.labels[-1].get_label_sz()[0]
	prms.labelNameStr = prms.labelNameStr[1:]
	if 'ptch' in labels or 'pose' in labels:
		prms.isSiamese = True
	else:
		prms.isSiamese = False

	prms['lbNrmlz'] = labelNrmlz
	prms['crpSz']   = crpSz
	prms['trnSeq']  = trnSeq
	prms.geoPoly    = None 	

	prms.splits = edict()
	prms.splits.numTrain = numTrain
	prms.splits.numTest  = numTest
	prms.splits.tePct    = tePct
	prms.splits.teGap    = teGap
	prms.splits.randSeed = 3

	#Form the splits file
	splitsStr = 'tePct%.1f_teGap%d_teSeed%d' % (tePct, teGap, prms.splits.randSeed) 
	paths.proc.splitsFile = paths.proc.splitsFile % (splitsStr + '/%s') 	
	splitDr, _ = osp.split(paths.proc.splitsFile)
	_mkdir(splitDr)

	expStr    = ''.join(['%s_' % lb.lbStr_ for lb in prms.labels])
	expStr    = expStr[0:-1]
	if geoFence is not None:
		expStr     = '%s_geo-%s' % (expStr, geoFence)
		paths.geoFile = paths.geoFile % geoFence
		prms.geoPoly  = read_geo_coordinates(paths.geoFile) 
	expName   = '%s_crpSz%d_nTr-%.2e' % (expStr, crpSz, numTrain)
	teExpName = '%s_crpSz%d_nTe-%.2e' % (expStr, crpSz, numTest)
	prms['expName'] = expName

	paths['windowFile'] = {}
	windowDir = paths.exp.window.dr
	paths['windowFile']['train'] = osp.join(windowDir, 'train_%s.txt' % expName)
	paths['windowFile']['test']  = osp.join(windowDir, 'test_%s.txt'  % teExpName)
	#paths['resFile']       = osp.join(paths['resDir'], expName, '%s.h5')

	prms['paths'] = paths
	#Get the pose stats
	prms['poseStats'] = {}
	#prms['poseStats']['mu'], prms['poseStats']['sd'], prms['poseStats']['scale'] =\
	#					get_pose_stats(prms)
	return prms

##
#Get normals for 
def get_prms_nrml(**kwargs):
	return get_prms(labels=['nrml'], labelType=['xyz'], lossType=['l2'], **kwargs)
		
def get_prms_ptch(**kwargs):
	return get_prms(labels=['ptch'], labelType=['wngtv'], lossType=['l2'], **kwargs)

def get_prms_pose(**kwargs):
	return get_prms(labels=['pose'], labelType=['quat'], lossType=['l2'], **kwargs)

def get_prms_pose_euler(**kwargs):
	return get_prms(labels=['pose'], labelType=['euler'], lossType=['l2'], **kwargs)
