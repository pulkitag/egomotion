## @package street_labels
#  Contains function for reading and parsing labels
#

#Self imports
import street_params as sp
import street_utils as su
#Other imports
import pickle
import numpy as np
from os import path as osp
import my_pycaffe_io as mpio
import subprocess
from multiprocessing import Pool
import copy
from pycaffe_config import cfg
import os
import pdb
from transforms3d.transforms3d import euler as t3eu

##
# Read the labels
def parse_label_file(fName):
	label = edict()
	with open(fName, 'r') as f:
		data = f.readlines()
		if len(data)==0:
			return None
		dl   = data[0].strip().split()
		assert dl[0] == 'd'
		label.ids    = edict()
		label.ids.ds = int(dl[1]) #DataSet id
		label.ids.tg = int(dl[2]) #Target id
		label.ids.im = int(dl[3]) #Image id
		label.ids.sv = int(dl[4]) #Street view id
		label.pts    = edict()
		label.pts.target = [float(n) for n in dl[5:8]] #Target point
		label.nrml   = [float(n) for n in dl[8:11]]
		#heading, pitch , roll
		#to compute the rotation matrix - use xyz' (i.e. first apply pitch, then 
		#heading and then the rotation about the camera in the coordinate system
		#of the camera - this is equivalent to zxy format.  
		label.pts.camera = [float(n) for n in dl[11:14]] #street view point not needed
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
#Get labels for normals
def get_label_nrml(prms, groups, numSamples, randSeed=1001):
	N = len(groups)
	oldState  = np.random.get_state()
	randState = np.random.RandomState(randSeed)
	lbs, prefix     = [], []
	perm1   = randState.choice(N,numSamples)
	#For label configuration 
	lbIdx   = prms.labelNames.index('nrml')
	lbInfo  = prms.labels[lbIdx]
	st,en   = prms.labelSzList[lbIdx], prms.labelSzList[lbIdx+1]
	if lbInfo.loss_ == 'classify':
		isClassify = True
	else:
		isClassify = False
	ptchFlag = False
	if 'ptch' in prms.labelNames:
		ptchFlag = True
		ptchIdx  = prms.labelNames.index('ptch')
		ptchLoc  = prms.labelSzList[ptchIdx]
	for p in perm1:
		try:
			gp  = groups[p]
		except:
			pdb.set_trace()
		idx = randState.permutation(gp.num)[0]
		#Ignore the last dimension as its always 0.
		lb        = np.zeros((prms.labelSz,)).astype(np.float32)
		st,en     = prms.labelSzList[0], prms.labelSzList[1]
		if not isClassify:
			en = en - 1
			lb[en]  = 1
			rawNrml = gp.data[idx].nrml[0:2]
			rawNrml = np.array([rawNrml[0], rawNrml[1], 0])
			cHead, cPitch = gp.data[idx].rots[0:2]
			cHead         = rot_range(cHead + 90)
			cHead, cPitch = (cHead * np.pi)/180.0, (cPitch * np.pi)/180.0
			rotMat        = ru.euler2mat(cHead, cPitch, isRadian=True)
			rotNrml       = np.dot(rotMat, rawNrml)
			#try:
			#	assert (np.abs(rotNrml[2]) < 1e-4)
			#except:
			#	pdb.set_trace()
			#nHead, nPitch = (nHead * 180.0)/np.pi, (nPitch * 180.0)/np.pi
			#nHead, nPitch = rot_range(nHead)/30.0, rot_range(nPitch)/30.0
			lb[st:en] = math.atan2(rotNrml[1], rotNrml[0]), math.asin(rotNrml[2])
			lb[st:en] = (lb[st:en] * 180.0)/np.pi
			lb[st]    = rot_range(lb[st])
			lb[st+1]  = rot_range(lb[st+1])
		else:
			nxBin  = su.find_bin_index(lbInfo.binRange_, gp.data[idx].nrml[0])
			nyBin  = su.find_bin_index(lbInfo.binRange_, gp.data[idx].nrml[1])
			lb[st:en] = nxBin, nyBin
		if ptchFlag:
			lb[ptchLoc] = 2
		lbs.append(lb)
		prefix.append((gp.folderId, gp.prefix[idx].strip(), None, None))
	np.random.set_state(oldState)		
	return lbs, prefix

##
def rot_range(rot):
	rot = np.mod(rot, 360)
	if rot > 180:
		rot = -(360 - rot)
	return rot

##
#Get the rotation labels
def get_rots_label(lbInfo, rot1, rot2, pt1=None, pt2=None):
	'''
		rot1, rot2: rotations in degrees
		pt1,  pt2 : the location of cameras expressed as (lat, long, height)
		the output labels are provided in radians
	'''
	y1, x1, z1 = map(lambda x: x*np.pi/180., rot1)
	rMat1      = t3eu.euler2mat(x1, y1, z1, 'szxy')
	y2, x2, z2 = map(lambda x: x*np.pi/180., rot2)
	rMat2      = t3eu.euler2mat(x2, y2, z2, 'szxy')
	dRot       = np.dot(rMat2, rMat1.transpose())
	#pitch, yaw, roll are rotations around x, y, z axis
	pitch, yaw, roll = t3eu.mat2euler(dRot, 'szxy')
	_, thRot  = t3eu.euler2axangle(pitch, yaw, roll, 'szxy')
	lb = None
	#Figure out if the rotation is within or outside the limits
	if lbInfo.maxRot_ is not None:
		if (np.abs(thRot) > lbInfo.maxRot_):
				return lb
	#Calculate the rotation
	if lbInfo.labelType_ == 'euler':
		if lbInfo.loss_ == 'classify':
			rollBin  = su.find_bin_index(lbInfo.binRange_, roll)
			yawBin   = su.find_bin_index(lbInfo.binRange_, yaw)
			pitchBin = su.find_bin_index(lbInfo.binRange_, pitch)
			if lbInfo.lbSz_ in [3,6]:
				lb = (rollBin, yawBin, pitchBin)
			else:
				lb = (yawBin, pitchBin)
		else:
			if lbInfo.lbSz_ == 3:
				lb = (pitch, yaw, roll)
			else:
				lb = (pitch, yaw)
	elif lbInfo.labelType_ == 'euler-5dof':
		dx, dy, dz = su.get_displacement_vector(pt1, pt2)
		lb = (pitch, yaw, dx, dy, dz )
	elif lbInfo.labelType_ == 'euler-6dof':
		dx, dy, dz = su.get_displacement_vector(pt1, pt2)
		lb = (pitch, yaw, roll, dx, dy, dz)
	elif lbInfo.labelType_ == 'quat':
		quat = t3eu.euler2quat(pitch, yaw, roll, axes='szxy')
		q1, q2, q3, q4 = quat
		lb = (q1, q2, q3, q4)
	else:
		raise Exception('Type not recognized')	
	return lb

##
#Get labels for pose
def get_label_pose(prms, groups, numSamples, randSeed=1003):
	#Remove the groups of lenght 1
	initLen = len(groups)
	groups = [g for g in groups if not(g.num==1)]
	N = len(groups)
	print ('Initial: %d, Final: %d' % (initLen, N))
	oldState  = np.random.get_state()
	randState = np.random.RandomState(randSeed)
	lbs, prefix = [], []
	lbCount = 0
	perm1   = randState.choice(N,numSamples)
	lbIdx   = prms.labelNames.index('pose')
	lbInfo  = prms.labels[lbIdx]
	st,en   = prms.labelSzList[lbIdx], prms.labelSzList[lbIdx+1]
	if lbInfo.loss_ == 'classify':
		isClassify = True
	else:
		isClassify = False
	if not isClassify:
		en = en - 1
	#Find if ptch matching is there and use the ignore label loss
	ptchFlag = False
	if 'ptch' in prms.labelNames:
		ptchFlag = True
		ptchIdx  = prms.labelNames.index('ptch')
		ptchLoc  = prms.labelSzList[ptchIdx]
	for p in perm1:
		lb  = np.zeros((prms.labelSz,)).astype(np.float32)
		if not isClassify:
			lb[en] = 1.0
		gp  = groups[p]
		lPerm  = randState.permutation(gp.num)
		n1, n2 = lPerm[0], lPerm[1]
		rotLb  = get_rots_label(lbInfo, gp.data[n1].rots, 
												gp.data[n2].rots,
                gp.data[n1].pts.camera, gp.data[n2].pts.camera)
		if rotLb is None:
			continue
		lb[st:en]  = rotLb
		prefix.append((gp.folderId, gp.prefix[n1].strip(),
							 gp.folderId, gp.prefix[n2].strip()))
		if ptchFlag:
			lb[ptchLoc] = 2
		lbs.append(lb)
	np.random.set_state(oldState)		
	return lbs, prefix

##
#Get labels for ptch
def get_label_ptch(prms, groups, numSamples, randSeed=1005):
	initLen = len(groups)
	groups = [g for g in groups if not(g.num==1)]
	N = len(groups)
	print ('Initial: %d, Final: %d' % (initLen, N))
	oldState  = np.random.get_state()
	randState = np.random.RandomState(randSeed)
	lbs, prefix = [],[]
	lbCount = 0
	perm1   = randState.choice(N,numSamples)
	perm2   = randState.choice(N,numSamples)
	lbIdx   = prms.labelNames.index('ptch')
	lbInfo  = prms.labels[lbIdx]
	st,en   = prms.labelSzList[lbIdx], prms.labelSzList[lbIdx+1]
	#Is sometimes needed
	poseLb = edict()
	poseLb.maxRot_ = prms.mxPtchRot
	poseLb.labelType_ = 'euler'
	poseLb.loss_ = 'l2'
	poseLb.lbSz_ = 2
	for p1, p2 in zip(perm1, perm2):
		lb  = np.zeros((prms.labelSz,)).astype(np.float32)
		prob   = randState.rand()
		if prob > lbInfo.posFrac_:
			#Sample positive
			lb[st] = 1
			gp  = groups[p1]
			n1  = randState.permutation(gp.num)[0]
			n2  = randState.permutation(gp.num)[0]
			if prms.mxPtchRot is not None:
				rotLbs     = get_rots_label(poseLb, gp.data[n1].rots, 
													 gp.data[n2].rots)
				if rotLbs is None:
					lb[st] = 2	
			prefix.append((gp.folderId, gp.prefix[n1].strip(),
										 gp.folderId, gp.prefix[n2].strip()))
		else:
			#Sample negative
			lb[st] = 0
			while (p1==p2):
				print('WHILE LOOP STUCK')
				p2 = (p1 + 1) % N
			gp1  = groups[p1]
			gp2  = groups[p2]
			n1  = randState.permutation(gp1.num)[0]
			n2  = randState.permutation(gp2.num)[0]
			prefix.append((gp1.folderId, gp1.prefix[n1].strip(),
										 gp2.folderId, gp2.prefix[n2].strip()))
		lbs.append(lb)
	return lbs, prefix

##
#Get labels for ptch
def get_label_pose_ptch(prms, groups, numSamples, randSeed=1005, randSeedAdd=0):
	initLen = len(groups)
	groups = [g for g in groups if not(g.num==1)]
	N = len(groups)
	print ('Initial: %d, Final: %d' % (initLen, N))
	oldState  = np.random.get_state()
	randState = np.random.RandomState(randSeed + randSeedAdd)
	lbs, prefix = [],[]
	lbCount = 0
	perm1   = randState.choice(N,numSamples)
	perm2   = randState.choice(N,numSamples)
	ptchIdx = prms.labelNames.index('ptch')
	poseIdx = prms.labelNames.index('pose')
	ptchLb  = prms.labels[ptchIdx]
	poseLb  = prms.labels[poseIdx]
	ptchSt,ptchEn   = prms.labelSzList[ptchIdx], prms.labelSzList[ptchIdx+1]
	poseSt,poseEn   = prms.labelSzList[poseIdx], prms.labelSzList[poseIdx+1]
	poseEn = poseEn - 1
	poseIdx, negIdx  = [], []
	idxCount          = 0
	for p1, p2 in zip(perm1, perm2):
		lb  = np.zeros((prms.labelSz,)).astype(np.float32)
		prob   = randState.rand()
		if prob < ptchLb.posFrac_:
			#Sample positive
			lb[ptchSt] = 1
			gp  = groups[p1]
			lPerm = randState.permutation(gp.num)
			n1, n2 = lPerm[0], lPerm[1]
			#Sample the pose as well
			rotLbs     = get_rots_label(poseLb, gp.data[n1].rots, 
													 gp.data[n2].rots)
			if rotLbs is None:
				continue
			lb[poseSt:poseEn] = rotLbs
			lb[poseEn] = 1.0
			mxAbs = max(np.abs(lb[poseSt:poseEn]))
			if prms.mxPtchRot is None:
				lb[ptchSt] = 1
			else:
				if prms.mxPtchRot >= mxAbs * 30:
					lb[ptchSt] = 1
				else:
					lb[ptchSt] = 2
			prefix.append((gp.folderId, gp.prefix[n1].strip(),
										 gp.folderId, gp.prefix[n2].strip()))
			poseIdx.append(idxCount)
			idxCount += 1
		else:
			#Sample negative
			lb[ptchSt] = 0
			while (p1==p2):
				print('WHILE LOOP STUCK')
				p2 = (p1 + 1) % N
			gp1  = groups[p1]
			gp2  = groups[p2]
			n1  = randState.permutation(gp1.num)[0]
			n2  = randState.permutation(gp2.num)[0]
			prefix.append((gp1.folderId, gp1.prefix[n1].strip(),
										 gp2.folderId, gp2.prefix[n2].strip()))
			negIdx.append(idxCount)
			idxCount += 1
		lbs.append(lb)

	posCount = len(poseIdx)
	negCount = len(negIdx)
	expectedNegCount = int(np.ceil((float(posCount)/ptchLb.posFrac_)*(1.0 - ptchLb.posFrac_)))
	if negCount > expectedNegCount:
		print ('%d pos, %d neg, but there should be %d neg, adjusting'\
							 % (posCount, negCount, expectedNegCount))
		perm = randState.permutation(negCount)
		permIdx = [negIdx[p] for p in perm]
		keepIdx = permIdx[0:expectedNegCount] + poseIdx
		#Randomly permute these indices
		perm    = randState.permutation(len(keepIdx))
		keepIdx = [keepIdx[p] for p in perm] 
		lbs = [lbs[k] for k in keepIdx]
		prefix = [prefix[k] for k in keepIdx] 
	else:	
		print ('%d negatives, there should be %d, NOT adjusting' % (negCount, expectedNegCount))	
	print ('#Labels Extracted: %d' % len(lbs))
	return lbs, prefix


def get_labels(prms, setName='train'):
	grps   = su.get_groups_all(prms, setName=setName)
	lbNums = []
	for (i,l) in enumerate(prms.labelNames):
		if l == 'nrml':
			lbNums.append(6 * len(grps))
		elif l == 'pose':
			lbNums.append(25 * len(grps))
		elif l == 'ptch':
			lbInfo = prms.labels[i]
			num = int(25.0 / lbInfo.posFrac_)
			lbNums.append(num * len(grps))			

	if prms.isMultiLabel:
		mxCount = max(lbNums)
		mxIdx   = lbNums.index(mxCount)
		if prms.labelNameStr == 'pose_ptch':
			lbs, prefix = get_label_pose_ptch(prms, grps, mxCount)
		else:
			raise Exception('%s multilabel not found' % prms.labelNameStr)
	else:
		assert(len(prms.labelNames)==1)
		lbName    = prms.labelNames[0]
		numSample = lbNums[0] 
		if lbName == 'nrml':
			lbs, prefix = get_label_nrml(prms, grps, numSample)
		elif lbName == 'pose':
			lbs, prefix = get_label_pose(prms, grps, numSample)
		elif lbName == 'ptch':
			lbs, prefix = get_label_ptch(prms, grps, numSample)
	return lbs, prefix	

##
#Get labels for all datapoints in folder with id 'folderid'
def get_label_by_folderid(prms, folderId, maxGroups=None):
	grpDict = su.get_groups(prms, folderId, setName=None)
	#Filter the ids if required
	if prms.splits.ver in ['v1']:
		#One folder either belongs to train or to test
		splitFile = prms.paths.proc.splitsFile % folderId
		splitDat  = pickle.load(open(splitFile, 'r'))
		splitDat  = splitDat['splits']
		if len(splitDat['train'])>0:
			assert(len(splitDat['test'])==0)
			gKeys = splitDat['train']
			print ('Folder: %s is TRAIN with %d Keys' % (folderId, len(gKeys)))
		else:
			assert(len(splitDat['train'])==0)
			gKeys = splitDat['test']
			print ('Folder: %s is TEST with %d Keys' % (folderId, len(gKeys)))
		grps = []
		for gk in gKeys:
			grps.append(grpDict[gk]) 
	else:		
		grps    = [g for (gk,g) in grpDict.iteritems()] 

	if maxGroups is not None and maxGroups < len(grps):
		perm = np.random.permutation(len(grps))
		perm = perm[0:maxGroups]
		grps = [grps[p] for p in perm]		

	#Form the Labels
	lbNums = []
	for (i,l) in enumerate(prms.labelNames):
		if l == 'nrml':
			lbNums.append(6 * len(grps))
		elif l == 'pose':
			lbNums.append(25 * len(grps))
		elif l == 'ptch':
			lbInfo = prms.labels[i]
			num = int(25.0 / lbInfo.posFrac_)
			lbNums.append(num * len(grps))			
	folderInt = int(folderId)
	if prms.isMultiLabel:
		mxCount = max(lbNums)
		mxIdx   = lbNums.index(mxCount)
		if prms.labelNameStr == 'pose_ptch':
			lbs, prefix = get_label_pose_ptch(prms, grps, mxCount, randSeedAdd=79 * folderInt)
		else:
			raise Exception('%s multilabel not found' % prms.labelNameStr)
	else:
		assert(len(prms.labelNames)==1)
		lbName    = prms.labelNames[0]
		numSample = lbNums[0] 
		if lbName == 'nrml':
			lbs, prefix = get_label_nrml(prms, grps, numSample)
		elif lbName == 'pose':
			lbs, prefix = get_label_pose(prms, grps, numSample)
		elif lbName == 'ptch':
			lbs, prefix = get_label_ptch(prms, grps, numSample)
	return lbs, prefix	

