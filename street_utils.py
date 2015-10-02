import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess

def _mkdir(fName):
	if not osp.exists(fName):
		os.makedirs(fName)

def process_folder():
	pass

##
# Helper function for get_foldernames
def _find_im_labels(folder):
	fNames = os.listdir(folder)
	fNames = [osp.join(folder, f.strip()) for f in fNames]
	#If one is directory then all should be dirs
	if osp.isdir(fNames[0]):
		dirNames = []
		for f in fNames:
			childDir = _find_im_labels(f)	
			if type(childDir) == list:
				dirNames = dirNames + childDir
			else:
				dirNames = dirNames + [childDir]
	else:
		dirNames = [folder]
	print dirNames
	return dirNames

##
# Find the foldernames in which data is stored
def get_foldernames(prms):
	'''
		Search for the folder tree - till the time there are no more
		directories. We will assume that the terminal folder has all
	  the needed files
	'''
	fNames = [f.strip() for f in os.listdir(prms.paths.raw.dr)]
	fNames = [osp.join(prms.paths.raw.dr, f) for f in fNames]
	print (fNames)
	fNames = [_find_im_labels(f) for f in fNames if osp.isdir(f)]
	return fNames

##
# Save the foldernames along with the folder ids
def save_foldernames(prms):	
	fNames = sorted(get_foldernames(prms))
	fid    = open(prms.paths.proc.folders.key, 'w')
	allNames = []
	for f in fNames:
		for ff in f:
			allNames  = allNames + [ff]
	allNames = sorted(allNames)
	for (i,f) in enumerate(allNames):
		fid.write('%04d \t %s\n' % (i + 1 ,f))
	fid.close()
	#For safety strip write permissions from the file
	subprocess.check_call(['chmod a-w %s' % prms.paths.proc.folders.key], shell=True) 

##
# Read names of all files in the folder
# Ensure that .jpg and .txt match and save those prefixes
def read_prefixes_from_folder(dirName):
	allNames = os.listdir(dirName)
	#Extract the prefixes
	imNames   = sorted([f for f in allNames if '.jpg' in f], reverse=True)
	lbNames   = sorted([f for f in allNames if '.txt' in f], reverse=True)
	prefixStr = []
	for (i,imn) in enumerate(imNames):
		imn = imn[0:-4]
		if i>= len(lbNames):
			continue
		if imn in lbNames[i]:
			prefixStr = prefixStr + [imn] 
	return prefixStr

##
# Save filename prefixes for each folder
def save_folder_prefixes(prms, forceWrite=False):
	fid   = open(prms.paths.proc.folders.key, 'r')
	lines = [l.strip() for l in fid.readlines()]
	fid.close()
	keys, names = [], []
	for l in lines:
		key, name = l.split()
		print (key, name)
		fName   = prms.paths.proc.folders.pre % key
		if osp.exists(fName) and (not forceWrite):
			continue
		preStrs = read_prefixes_from_folder(name) 
		with open(fName, 'w') as f:
			for p in preStrs:
				f.write('%s \n' % p)
		

def get_tar_files(prms):
	with open(prms.paths.tar.fileList,'r') as f:
		fNames = f.readlines()
	fNames = [f.strip() for f in fNames]
	return fNames

def download_tar(prms):
	fNames = get_tar_files(prms)
	for f in fNames:
		_, name = osp.split(f)
		outName = osp.join(prms.paths.tar.dr, name)
		print(outName)
		if not osp.exists(outName):
			print ('Downloading')
			subprocess.check_call(['gsutil cp %s %s' % (f, outName)], shell=True) 
	print ('All Files copied')
	

def get_paths():
	paths      = edict()
	#For storing the directories
	paths.dirs = edict()
	#The raw data
	paths.dataDr  = '/data0/pulkitag/data_sets/streetview'
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
	paths.code.dr = '/home/ubuntu/code/streetview'
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

	#Label data
	paths.label    = edict()
	paths.label.dr   = osp.join(paths.proc.dr, 'labels')
	nrmlDir          = osp.join(paths.label.dr, 'nrml')
	_mkdir(nrmlDir)
	paths.label.nrml = osp.join(nrmlDir, '%s.txt')		 

	#Window data file
	paths.exp    = edict()
	paths.exp.dr = osp.join(paths.dataDr, 'exp')
	_mkdir(paths.exp.dr)
	paths.exp.window    = edict()
	paths.exp.window.dr = osp.join(paths.exp.dr, 'window-files')
	_mkdir(paths.exp.window.dr) 
	paths.exp.window.tr = osp.join(paths.exp.window.dr, 'train-%s.txt')
	paths.exp.window.te = osp.join(paths.exp.window.dr, 'test-%s.txt')
	
	return paths

def get_data_files(prms):
	allNames = os.listdir(prms.paths.dirs.rawData)
	#Extract the prefixes
	imNames   = sorted([f for f in allNames if '.jpg' in f], reverse=True)
	lbNames   = sorted([f for f in allNames if '.txt' in f], reverse=True)
	prefixStr = []
	for (i,imn) in enumerate(imNames):
		imn = imn[0:-4]
		if imn in lbNames[i]:
			prefixStr = prefixStr + [imn] 
	imNames = [f + '.jpg' for f in prefixStr]
	lbNames = [f + '.txt' for f in prefixStr]
	return imNames, lbNames

##
# Get the prms 
def get_prms(isAligned=True):
	prms = edict()
	prms.paths = get_paths()
	prms.isAligned = isAligned
	return prms

##
# Get the label dimensions
def get_label_size(labelClass, labelType):
	if labelClass == 'nrml':
		if labelType == 'xyz':
			lSz = 3
		else:
			raise Exception('%s,%s not recognized' % (labelClass, labelType))
	elif labelClass == 'ptch':
		if labelType in ['wngtv', 'hngtv']:
			lSz = 3
		else:
			raise Exception('%s,%s not recognized' % (labelClass, labelType))
	elif labelClass == 'pose':
		if labelType in ['quat', 'euler']:
			lSz = 6
		else:
			raise Exception('%s,%s not recognized' % (labelClass, labelType))
	else:
		raise Exception('%s not recognized' % labelClass)
	return lSz

##
#get prms
def get_prms_v2(labels=['nrml'], nrmlType ='xyz',
						 ptchType = 'wngtv', 
						 poseType='quat', labelNrmlz=None, 
						 crpSz=256,
						 numTrain=1e+06, numTest=1e+04,
						 lossType='L2',
						 randomCrop=True, trnSeq=[]):
	'''
		labels    : What labels to use - make it a list for multiple
								 kind of labels
								 nrml - surface normals
								 ptch - patch matching
		nrmlType   : How is normal data represented
								 xyz - as nx, ny, nz
		ptchType   : How is patch data represented
								 wngtv - weak negatives 
								 hngtv = hard negatices
		poseType   : How pose is being used.
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
		concatLayer: The layer used for concatentation in siamese training
		randomCrop  : Whether to randomly crop the images or not. 	
		trnSeq      : Manually specif train-sequences by hand

		NOTES
		randomCrop, concatLayer are properties of the training
                            they should not be in prms, but in caffePrms
	'''
	assert type(labels) == list, 'labelType must be a list'

	paths = get_paths()
	prms  = edict()
	prms.labels     = labels
	prms.ltype      = edict()
	prms.ltype.nrml = nrmlType
	prms.ltype.ptch = ptchType
	prms.ltype.pose = poseType
	prms['lbNrmlz'] = labelNrmlz
	prms['imSz']         = imSz
	prms['concatLayer']  = concatLayer  
	prms['lossType']     = lossType
	prms['randomCrop']   = randomCrop
	prms['trnSeq']       = trnSeq

	prms.numSamples = edict()
	prms.numSamples.train = numTrain
	prms.numSamples.test  = numTest

	#Comute the labelSz and labelStr
	labelSz, labelStr = 0, ''
	for l in sorted(labels):
		labelStr = labelStr + '%s-%s-' % (l, prms.ltype[l]) 
		labelSz  = labelSz + get_label_size(l, prms.ltype[l])
	labelStr = labelStr[0:-1]

	#This should be part of caffe_prms too
	lossStr = ''
	if lossType in ['l2', 'l1', 'l2-tukey', 'cntrstv']:
		lossStr = 'loss-%s' % lossType
	else:
		raise Exception('%s loss type not recognized' % lossStr)

	trainExpStr = '%s_%s'% (labelStr, lossStr)
	testExpStr  = labelStr	
	expName   = 'mxDiff-%d_pose-%s_nrmlz-%s_%s_concat-%s_nTr-%d'\
							 % (maxFrameDiff, poseType, nrmlzType, imStr, concatLayer, numTrainSamples) 
	teExpName =  'mxDiff-%d_pose-%s_nrmlz-%s_%s_concat-%s_nTe-%d'\
							 % (maxFrameDiff, poseType, nrmlzType, imStr, concatLayer, numTestSamples) 
	expName   = expStr + expName
	teExpname = expStr + teExpName 

	prms['expName'] = expName

	paths['windowFile'] = {}
	paths['windowFile']['train'] = os.path.join(paths['windowDir'], 'train_%s.txt' % expName)
	paths['windowFile']['test']  = os.path.join(paths['windowDir'], 'test_%s.txt'  % teExpName)
	paths['resFile']       = os.path.join(paths['resDir'], expName, '%s.h5')

	prms['paths'] = paths
	#Get the pose stats
	prms['poseStats'] = {}
	prms['poseStats']['mu'], prms['poseStats']['sd'], prms['poseStats']['scale'] =\
						get_pose_stats(prms)
	return prms



##
# Get key for all the folders
def get_folder_keys_all(prms):
	allKeys, allNames = [], []
	with open(prms.paths.proc.folders.key, 'r') as f:
		lines = f.readlines()
		for l in lines:
			key, name = l.strip().split()
			allKeys.append(key)
			allNames.append(name)
	return allKeys, allNames 

##
# Save the key of the folders that are aligned. 
def get_folder_keys_aligned(prms):
	with open(prms.paths.proc.folders.aKey,'r') as f:
		keys = f.readlines()
		keys = [k.strip() for k in keys]
	return keys		

##
# Save the keys of only the folders for which alignment data is available. 
def save_aligned_keys(prms):
	keys, names = get_folder_keys_all(prms)
	with open(prms.paths.proc.folders.aKey,'w') as f:
		for k,n in zip(keys,names):	
			if 'Aligned' in n:
				f.write('%s\n' % k)

##
#Return the name of the folder from the id
def id2name_folder(prms, folderId):
	outName = None
	with open(prms.paths.proc.folders.key, 'r') as f:
		lines = f.readlines()
		for l in lines:
			key, name = l.strip().split()
			if key == folderId:
				outName = name	
	return outName

##
# Get the names and labels of the files of a certain id 
def folderid_to_im_label_files(prms, folderId):
	with open(prms.paths.proc.folders.pre % folderId,'r') as f:
		prefixes = f.readlines()
		folder   = id2name_folder(prms, folderId)
		imNames, lbNames = [], []
		for p in prefixes:
			imNames.append(osp.join(folder, '%s.jpg' % p.strip()))
			lbNames.append(osp.join(folder, '%s.txt' % p.strip()))
	return imNames, lbNames	

##
# Read the labels
def parse_label_file(fName):
	label = edict()
	with open(fName, 'r') as f:
		data = f.readlines()
		dl   = data[0].strip().split()
		assert dl[0] == 'd'
		label.ids    = edict()
		label.ids.ds = int(dl[1]) #DataSet id
		label.ids.tg = int(dl[2]) #Target id
		label.ids.im = int(dl[3]) #Image id
		label.ids.sv = int(dl[4]) #Street view id
		#dl[5,6,7] -- target point will skip
		label.nrml   = [float(n) for n in dl[8:11]]
		#dl[11,12,13] -- street view point not needed
		label.dist   = float(dl[14])
		label.rots   = [float(n) for n in dl[15:18]]
		assert label.rots[2] == 0, 'Something is wrong %s' % fName	
		label.align = None
		if len(data) == 2:
			al = data[1].strip().split()[1:]
			label.align = edict()
			#Corrected patch center
			label.align.loc	 = [float(n) for n in al[0:2]]
			#Warp matrix	
			label.align.warp = np.array([float(n) for n in al[2:11]])
	return label
		
##
#Save the normal data
def save_normals(prms):
	if prms.isAligned:
		ids = get_folder_keys_aligned(prms)
	else:
		ids = get_folder_keys_all(prms)
	for i in ids:
		count = 0
		imFiles, labelFiles = folderid_to_im_label_files(prms,i)
		with open(prms.paths.label.nrml % i, 'w') as fid:
			for (imf,lbf) in zip(imFiles,labelFiles):
				if np.mod(count,1000)==1:
					print(count)
				lb = parse_label_file(lbf)
				_, imfStr = osp.split(imf)
				fid.write('%s \t %f \t %f \t %f\n' % (imfStr,lb.nrml[0],
											lb.nrml[1], lb.nrml[2]))
				count += 1
	
 
