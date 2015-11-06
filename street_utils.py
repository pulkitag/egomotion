## @package street_utils
#
#

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
import my_pycaffe_io as mpio
import re
import matplotlib.path as mplPath
import rot_utils as ru
from geopy.distance import vincenty as geodist
import copy
import street_params as sp
from multiprocessing import Pool

##
#helper functions
def find_first_false(idx):
	for i in range(len(idx)):
		if not idx[i]:
			return i
	return None

##
#Find the bin index
def find_bin_index(bins, val):
	idx = find_first_false(val>=bins)
	if idx==0:
		print ('WARNING - CHECK THE RANGE OF VALUES - %f was BELOW THE MIN' % val)
	if idx is None:
		return len(bins)-2
	return max(0,idx-1)

##
#Get the prefixes for a specific folderId
def get_prefixes(prms, folderId):
	with open(prms.paths.proc.folders.pre % folderId,'r') as f:
		prefixes = [p.strip() for p in f.readlines()]
	return prefixes

##
#Get the groups of images that are taken by pointing at the same
#target location
def get_target_groups(prms, folderId):
	prefixes = get_prefixes(prms, folderId)
	S        = []
	prev     = None
	count    = 0
	for (i,p) in enumerate(prefixes):	
		_,_,_,grp = p.split('_')
		if not(grp == prev):
			S.append(count)
			prev = grp
		count += 1
	return S

##
#Get the distance between groups, 
#Finds the minimum distance between as 
# min(dist_camera_points, dist_target_point)
def get_distance_groups(grp1, grp2):
	minDist = np.inf
	for n1 in range(grp1.num):
		cpt1 = grp1.data[n1].pts.camera[0:2]
		tpt1 = grp1.data[n1].pts.target[0:2]
		for n2 in range(grp2.num):	
			cpt2 = grp2.data[n2].pts.camera[0:2]
			tpt2 = grp2.data[n2].pts.target[0:2]
			cDist = geodist(cpt1, cpt2).meters
			tDist = geodist(tpt1, tpt2).meters
			dist  = min(cDist, tDist)
			if dist < minDist:
				minDist = dist
	return minDist

##
#Seperate points based on the target distance.
def get_distance_targetpts_groups(grp1, grp2):
	tPt1 = grp1.data[0].pts.target
	tPt2 = grp2.data[0].pts.target
	tDist = geodist(tPt1, tPt2).meters
	return tDist

##
#Get the distance between lists of groups
def get_distance_grouplists(grpList1, grpList2):
	minDist = np.inf
	for g1 in grpList1:
		for g2 in grpList2:
			dist = get_distance_groups(g1, g2)
			if dist < minDist:
					minDist = dist
	return minDist	
	
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
# Get the key of the folders that are aligned. 
def get_folder_keys_aligned(prms):
	with open(prms.paths.proc.folders.aKey,'r') as f:
		keys = f.readlines()
		keys = [k.strip() for k in keys]
	return keys		

##
#Get the keys for a folder
def get_folder_keys(prms):
	keys,_   = get_folder_keys_all(prms)
	return keys

##
#Get the list of all folders
def get_folder_list(prms):
	allKeys, allNames = get_folder_keys_all(prms)
	fList = edict()
	for (k,n) in zip(allKeys, allNames):
		fList[k] = n
	if prms.isAligned:
		aKeys = get_folder_keys(prms)
		for k in fList.keys():
			if k not in aKeys:
				del fList[k]
	return fList	

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
def folderid_to_im_label_files(prms, folderId, opPrefix=False):
	with open(prms.paths.proc.folders.pre % folderId,'r') as f:
		prefixes = f.readlines()
		folder   = id2name_folder(prms, folderId)
		imNames, lbNames = [], []
		for p in prefixes:
			imNames.append(osp.join(folder, '%s.jpg' % p.strip()))
			lbNames.append(osp.join(folder, '%s.txt' % p.strip()))
	if opPrefix:
		return imNames, lbNames, prefixes
	else:
		return imNames, lbNames	

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
#Get the overall count of number of groups in the dataset
def get_group_counts(prms):
	dat = pickle.load(open(prms.paths.proc.countFile, 'r'))
	if prms.isAligned:
		keys   = get_folder_keys_aligned(prms)	
	else:
		keys,_ = get_folder_keys_all(prms)	
	count = 0
	for k in keys:
		count += dat['groupCount'][k]
	return count
		
##
#Get the train and test splits
def get_train_test_splits(prms, folderId):
	fName  = prms.paths.proc.splitsFile % folderId
	splits = edict(pickle.load(open(fName,'r')))
	return splits.splits

##
#polygon of type mplPath
def is_geo_coord_inside(polygon,cord):
	return polygon.contains_point(cord)


##
#Find if a group is inside the geofence
def is_group_in_geo(prms, grp):
	isInside = False
	if prms.geoPoly is None:
		return True
	else:
		#Even if a single target point is inside the geo
		#fence count as true
		for geo in prms.geoPoly:
			for i in range(grp.num):
				cc = grp.data[i].pts.target
				isInside = isInside or is_geo_coord_inside(geo, (cc[1], cc[0]))	
	return isInside

##
#Read Geo groups
def read_geo_groups_all(prms):
	geoGrps = edict()
	keys    = get_folder_keys(prms)
	for k in keys:
		geoGrps[k] = read_geo_groups(prms, k)
	return geoGrps

##
#Read geo group from a particular folder
def read_geo_groups(prms, folderId):
	fName      = prms.paths.grp.geoFile % folderId
	data       = pickle.load(open(fName,'r'))
	return data['groups']

##
#Get geo folderids
def get_geo_folderids(prms):
	if prms.geoFence in ['dc-v2', 'cities-v1', 'vegas-v1']:
		keys = []
		with open(prms.paths.geoFile,'r') as fid:
			lines = fid.readlines()
			for l in lines:
				key, _ = l.strip().split()
				keys.append(key)	
	else:
		raise Exception('Not found')
	return keys

##
#Get the groups
def get_groups(prms, folderId, setName='train'):
	'''
		Labels for a particular split
	'''
	grpList   = []
	if prms.geoFence in ['dc-v2', 'cities-v1', 'vegas-v1']:
		keys = get_geo_folderids(prms)
		if folderId not in keys:
			return grpList
		
	if prms.geoFence == 'dc-v1':
		groups = read_geo_groups(prms, folderId)
		gKeys  = groups.keys()
	else:
		#Read labels from the folder
		if prms.isAligned:  
			grpFile = prms.paths.label.grpsAlgn % folderId
		else:
			grpFile = prms.paths.label.grps % folderId
		grpData = pickle.load(open(grpFile,'r'))
		groups  = grpData['groups']
		gKeys   = groups.keys()

	if setName is not None:
		#Find the groups belogning to the split
		splits    = get_train_test_splits(prms, folderId)
		gSplitIds = splits[setName]
		for g in gSplitIds:
			if g in gKeys:
				grpList.append(groups[g])
		return grpList
	else:
		return copy.deepcopy(groups)

##
#Get all the raw labels
def get_groups_all(prms, setName='train'):
	keys = get_folder_keys(prms)
	#keys  = ['0052']
	grps   = []
	for k in keys:
		grps = grps + get_groups(prms, k, setName=setName)
	return grps

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
			lb[st:en] = gp.data[idx].nrml[0:2]
		else:
			nxBin  = find_bin_index(lbInfo.binRange_, gp.data[idx].nrml[0])
			nyBin  = find_bin_index(lbInfo.binRange_, gp.data[idx].nrml[1])
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
def get_rots_label(lbInfo, rot1, rot2):
	y1, x1, z1 = rot1
	y2, x2, z2 = rot2
	roll, yaw, pitch = rot_range(z2 - z1), rot_range(y2 - y1), rot_range(x2 - x1)
	lb = None
	#Figure out if the rotation is within or outside the limits
	if lbInfo.maxRot_ is not None:
		if (np.abs(roll) > lbInfo.maxRot_ or\
				np.abs(yaw) > lbInfo.maxRot_ or\
				np.abs(pitch)>lbInfo.maxRot_):
				return lb
	#Calculate the rotation
	if lbInfo.labelType_ == 'euler':
		if lbInfo.loss_ == 'classify':
			rollBin  = find_bin_index(lbInfo.binRange_, roll)
			yawBin   = find_bin_index(lbInfo.binRange_, yaw)
			pitchBin = find_bin_index(lbInfo.binRange_, pitch)
			if lbInfo.lbSz_  == 3:
				lb = (rollBin, yawBin, pitchBin)
			else:
				lb = (yawBin, pitchBin)
		else:
			if lbInfo.lbSz_ == 3:
				lb = (roll/30.0, yaw/30.0, pitch/30.0)
			else:
				lb = (yaw/30.0, pitch/30.0)
	elif lbInfo.labelType_ == 'quat':
		quat = ru.euler2quat(z2-z1, y2-y1, x2-x1, isRadian=False)
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
												gp.data[n2].rots)
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
	grps   = get_groups_all(prms, setName=setName)
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

		'''
		##### OLD CODE ######
		allLb, allPrefix = [], []
		allSamples = 0
		for (i,l)	in enumerate(prms.labelNames):
			numSample = (prms.labelFrac[i]/prms.labelFrac[mxIdx])*float(mxCount)
			print (i, l, numSample)
			if l== 'nrml':
				lbs, prefix = get_label_nrml(prms, grps, numSample)
			elif l == 'pose':
				lbs, prefix = get_label_pose(prms, grps, numSample)
			elif l == 'ptch':
				lbs, prefix = get_label_ptch(prms, grps, numSample)
			assert len(lbs)==numSample, '%d, %d' % (len(lbs), numSample)
			allSamples = int(allSamples + numSample)
			allLb     = allLb + lbs
			allPrefix = allPrefix + prefix 
		print ("Set: %s, total Number of Samples is %d" % (setName,allSamples))	
		perm = np.random.permutation(allSamples)
		lbs    = [allLb[p] for p in perm]
		prefix = [allPrefix[p] for p in perm] 	
		'''
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
# Convert a prefix and folder into the image name
def prefix2imname(prms, prefixes):
	fList   = get_folder_list(prms)
	for ff in fList.keys():
		drName    = fList[ff].split('/')[-1]
		fList[ff] = drName
	#print fList
	imNames = []
	for pf in prefixes:
		f1, p1, f2, p2 = pf
		if f2 is not None:
			imNames.append([osp.join(fList[f1], p1+'.jpg'), osp.join(fList[f2], p2 +'.jpg')])
		else:
			imNames.append([osp.join(fList[f1], p1+'.jpg'), None])
	return imNames

##
#Convert prefixes to image name
def prefix2imname_geo(prms, prefixes):
	if prms.geoFence=='dc-v1':
		keyData = pickle.load(open(prms.paths.proc.im.keyFile, 'r'))
		imKeys  = keyData['imKeys']
		imNames = []
		for pf in prefixes:
			f1, p1, f2, p2 = pf
			if f2 is not None:
				imNames.append([imKeys[f1][p1], imKeys[f2][p2]])
			else:
				imNames.append([imKeys[f1][p1], None])
	elif prms.geoFence in ['dc-v2', 'cities-v1']:
		raise Exception('Doesnot work for %s', prms.geoFence)
	return imNames
			
##
#Make the window files
def make_window_file(prms, setNames=['test', 'train']):
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

	for s in setNames:
		#Get the im-label data
		lb, prefix = get_labels(prms, s)
		if prms.geoFence is None:	
			imNames1 = prefix2imname(prms, prefix)
		else:
			imNames1 = prefix2imname_geo(prms, prefix) 
		#Randomly permute the data
		N = len(imNames1)
		randState = np.random.RandomState(19)
		perm      = randState.permutation(N) 
		#The output file
		gen = mpio.GenericWindowWriter(prms['paths']['windowFile'][s],
						len(imNames1), numImPerExample, prms['labelSz'])
		for i in perm:
			line = []
			for n in range(numImPerExample):
				line.append([imNames1[i][n], [ch, h, w], [minW, minH, maxW, maxH]])
			gen.write(lb[i], *line)
		gen.close()


def get_label_by_folderid(prms, folderId, maxGroups=None):
	grpDict = get_groups(prms, folderId, setName=None)
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
	lb, prefix = get_label_by_folderid(prms, folderId, maxGroups=maxGroups)
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
	keys   = get_geo_folderids(prms)
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
	assert (pCum==1, 'Something is wrong')
	randState = np.random.RandomState(31)
	ignoreCount = 0
	
	nrmlPrune = False
	if 'nrml' in prms.labelNames and len(prms.labelNames)==1:
		if prms.nrmlMakeUni is not None:
			idx = prms.labelNames.index('nrml')
			lbInfo = prms.labels[idx]
			nrmlPrune = True
			if lbInfo.loss_ in ['l2', 'l1']:
				nrmlBins  = np.linspace(-1,1,101)
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
#Fetch the window file from the main machine
def fetch_window_file_scp(prms):
	setNames = ['train', 'test']
	hostName = 'ubuntu@54.173.41.3:/data0/pulkitag/data_sets/streetview/exp/window-files/'
	for s in setNames:
		wName      = prms['paths']['windowFile'][s]
		_, fName   = osp.split(wName)
		remoteName = hostName + fName
		scpCmd = 'scp -i "pulkit-key.pem" '
		localName = prms['paths']['windowFile'][s]
		subprocess.check_call(['%s %s %s' % (scpCmd, remoteName, localName)],shell=True) 

def fetch_cropim_tar_by_folderid(args):
	prms, folderId = args
	hostName = 'ubuntu@54.173.41.3:/data0/pulkitag/data_sets/streetview/proc/resize-im/im256/'
	trFile = prms.paths.proc.im.folder.tarFile % folderId
	remoteName = hostName + '%s.tar' % folderId
	scpCmd = 'scp -i "pulkit-key.pem" '
	localName = trFile
	subprocess.check_call(['%s %s %s' % (scpCmd, remoteName, localName)],shell=True) 


#Send the window file to a host
def send_window_file_scp(prms, setNames=None):
	if setNames is None:
		setNames = ['train', 'test']
	hostName = 'pulkitag@psi.millennium.berkeley.edu:/work5/pulkitag/data_sets/streetview/'
	for s in setNames:
		wName      = prms['paths']['windowFile'][s]
		_, fName   = osp.split(wName)
		remoteName = hostName + fName
		scpCmd = 'scp '
		localName = prms['paths']['windowFile'][s]
		subprocess.check_call(['%s %s %s' % (scpCmd, localName, remoteName)],shell=True) 
 
##
#Process the normals prototxt
def process_normals_proto(prms, setName='test'):
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
#Convert a pose patch window file into a window file for patch matching only
def convert_pose_ptch_2_ptch(inFile, outFile):
	inFid  = mpio.GenericWindowReader(inFile)
	outFid = mpio.GenericWindowWriter(outFile, inFid.num_, 2, 1)
	while not inFid.is_eof():
		imData, lb = inFid.read_next()
		lbls = [[lb[0][2]]]	
		outFid.write(lbls[0], *imData)
	inFid.close()
	outFid.close()


'''
##
#Process the labels according to prms
def get_labels_old(prms, setName="train"):
	#The main quantity that requires randomization is patch matching
	#So we will base this code around that. 
	rawLb = get_groups_all(prms, setName=setName)
	N  = len(rawLb)
	oldState  = np.random.get_state()
	randSeed  = 1001
	randState = np.random.RandomState(randSeed)
	perm1     = randState.permutation(N)
	perm2     = randState.permutation(N)	
	perms     = zip(perm1,perm2)
	#get the labels
	lb, prefix = [], []
	for (i, perm) in enumerate(perms):
		p1, p2 = perm
		for lbType in prms.labels:
			if lbType.label_ == 'nrml':
				#1 because we are going to have this as input to the
				# ignore euclidean loss layer
				rl = rawLb[p1]
				for i in range(rl.num):
					#Ignore the last dimension as its always 0.
					lb.append(rl.data[i].nrml[0:2])
					prefix.append((rl.folderId, rl.prefix[i].strip(), None, None))
			elif lbType.label_ == 'ptch':
				#Based on the idea that there are typically 5 samples per group
				numRep = int(5.0 / lbType.posFrac_)
				for rep in range(numRep):
					prob   = randState.rand()
					p1, p2 = randState.random_integers(N-1), randState.random_integers(N-1)
					rl1  = rawLb[p1]
					rl2  = rawLb[p2]
					localPerm1 = randState.permutation(rl1.num)
					localPerm2 = randState.permutation(rl2.num)
					if prob > lbType.posFrac_:
						#Sample positive
						lb.append([1])	
						prefix.append((rl1.folderId, rl1.prefix[localPerm1[0]].strip(),
													 rl1.folderId, rl1.prefix[localPerm1[1]].strip()))
					else:
						#Sample negative			
						lb.append([0])
						prefix.append((rl1.folderId, rl1.prefix[localPerm1[0]].strip(),
													 rl2.folderId, rl2.prefix[localPerm2[0]].strip()))
			elif lbType.label_ == 'pose':
				rl1        = rawLb[p1]
				for n1 in range(rl1.num):
					for n2 in range(n1+1, rl1.num):
						if rl1.data[n1].align is None or rl1.data[n2].align is None:
							continue	 
						y1, x1, z1 = rl1.data[n1].rots
						y2, x2, z2 = rl1.data[n2].rots
						roll, yaw, pitch = z2 - z1, y2 - y1, x2 - x1
						if lbType.maxRot_ is not None:
							if (np.abs(roll) > lbType.maxRot_ or\
								  np.abs(yaw) > lbType.maxRot_ or\
								  np.abs(pitch)>lbType.maxRot_):
									continue
						if lbType.labelType_ == 'euler':
							if lbType.lbSz_ == 3:
								lb.append([roll/180.0, yaw/180.0, pitch/180.0]) 
							else:
								lb.append([yaw/180.0, pitch/180.0]) 
						elif lbType.labelType_ == 'quat':
							quat = ru.euler2quat(z2-z1, y2-y1, x2-x1, isRadian=False)
							q1, q2, q3, q4 = quat
							lb.append([q1, q2, q3, q4]) 
						else:
							raise Exception('Type not recognized')	
						prefix.append((rl1.folderId, rl1.prefix[n1].strip(),
													 rl1.folderId, rl1.prefix[n2].strip()))
			else:
				raise Exception('Type not recognized')	
	np.random.set_state(oldState)		
	return lb, prefix					

'''
