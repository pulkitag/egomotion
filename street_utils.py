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
import my_pycaffe_io as mpio
import re
import matplotlib.path as mplPath
import rot_utils as ru
from geopy.distance import vincenty as geodist
import copy

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
#Get the distance between groups
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
	if prms.isAligned:
		keys = get_folder_keys_aligned(prms)
	else:
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
		geoGrps[k] = read_geo_group(prms, k)
	return geoGrps

##
#Read geo group from a particular folder
def read_geo_groups(prms, folderId):
	fName      = prms.paths.grp.geoFile % folderId
	data       = pickle.load(open(fName,'r'))
	return data['groups']

##
#Get the groups
def get_groups(prms, folderId, setName='train'):
	'''
		Labels for a particular split
	'''
	grpList   = []
	if prms.geoFence is not None:
		groups = read_geo_groups(prms, folderId)
		gKeys  = groups.keys()
	else:
		#Read labels from the folder 
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
	st,en   = prms.labelSzList[lbIdx], prms.labelSzList[lbIdx+1]
	en = en - 1
	ptchFlag = False
	if 'ptch' in prms.labelNames:
		ptchFlag = True
		ptchIdx  = prms.labelNames.index('ptch')
		ptchLoc  = prms.labelSzList[ptchIdx]
	for p in perm1:
		gp  = groups[p]
		idx = randState.permutation(gp.num)[0]
		#Ignore the last dimension as its always 0.
		lb        = np.zeros((prms.labelSz,)).astype(np.float32)
		st,en     = prms.labelSzList[0], prms.labelSzList[1]
		lb[st:en] = gp.data[idx].nrml[0:2]
		lb[en]  = 1
		if ptchFlag:
			lb[ptchLoc] = 2
		lbs.append(lb)
		prefix.append((gp.folderId, gp.prefix[idx].strip(), None, None))
	np.random.set_state(oldState)		
	return lbs, prefix

##
#Get labels for pose
def get_label_pose(prms, groups, numSamples, randSeed=1003):
	N = len(groups)
	oldState  = np.random.get_state()
	randState = np.random.RandomState(randSeed)
	lbs, prefix = [], []
	lbCount = 0
	perm1   = randState.choice(N,numSamples)
	lbIdx   = prms.labelNames.index('pose')
	lbInfo  = prms.labels[lbIdx]
	st,en   = prms.labelSzList[lbIdx], prms.labelSzList[lbIdx+1]
	en = en - 1
	#Find if ptch matching is there and use the ignore label loss
	ptchFlag = False
	if 'ptch' in prms.labelNames:
		ptchFlag = True
		ptchIdx  = prms.labelNames.index('ptch')
		ptchLoc  = prms.labelSzList[ptchIdx]
	for p in perm1:
		lb  = np.zeros((prms.labelSz,)).astype(np.float32)
		lb[en] = 1.0
		gp  = groups[p]
		lPerm  = randState.permutation(gp.num)
		n1, n2 = lPerm[0], lPerm[1]
		y1, x1, z1 = gp.data[n1].rots
		y2, x2, z2 = gp.data[n2].rots
		roll, yaw, pitch = z2 - z1, y2 - y1, x2 - x1
		if lbInfo.maxRot_ is not None:
			if (np.abs(roll) > lbInfo.maxRot_ or\
					np.abs(yaw) > lbInfo.maxRot_ or\
					np.abs(pitch)>lbInfo.maxRot_):
					continue
		if lbInfo.labelType_ == 'euler':
			if lbInfo.lbSz_ == 3:
				lb[st:en] = roll/180.0, yaw/180.0, pitch/180.0
			else:
				lb[st:en] = yaw/180.0, pitch/180.0
		elif lbInfo.labelType_ == 'quat':
			quat = ru.euler2quat(z2-z1, y2-y1, x2-x1, isRadian=False)
			q1, q2, q3, q4 = quat
			lb[st:en] = q1, q2, q3, q4
		else:
			raise Exception('Type not recognized')	
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
	N = len(groups)
	oldState  = np.random.get_state()
	randState = np.random.RandomState(randSeed)
	lbs, prefix = [],[]
	lbCount = 0
	perm1   = randState.choice(N,numSamples)
	perm2   = randState.choice(N,numSamples)
	lbIdx   = prms.labelNames.index('ptch')
	lbInfo  = prms.labels[lbIdx]
	st,en   = prms.labelSzList[lbIdx], prms.labelSzList[lbIdx+1]
	for p1, p2 in zip(perm1, perm2):
		lb  = np.zeros((prms.labelSz,)).astype(np.float32)
		prob   = randState.rand()
		if prob > lbInfo.posFrac_:
			#Sample positive
			lb[st] = 1
			gp  = groups[p1]
			n1  = randState.permutation(gp.num)[0]
			n2  = randState.permutation(gp.num)[0]
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
	keyData = pickle.load(open(prms.paths.proc.im.keyFile, 'r'))
	imKeys  = keyData['imKeys']
	imNames = []
	for pf in prefixes:
		f1, p1, f2, p2 = pf
		if f2 is not None:
			imNames.append([imKeys[f1][p1], imKeys[f2][p2]])
		else:
			imNames.append([imKeys[f1][p1], None])
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


##
#Process the labels according to prms
def get_labels_old(prms, setName='train'):
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


