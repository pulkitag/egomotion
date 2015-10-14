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
#Get the raw labels
def get_raw_labels(prms, folderId, setName='train'):
	'''
		Labels for a particular split
	'''
	#Find the groups belogning to the split
	splits = get_train_test_splits(prms, folderId)
	gids   = splits[setName]
	#Read labels from the folder 
	lbFile = prms.paths.label.grps % folderId
	lbData = pickle.load(open(lbFile,'r'))
	lbData = lbData['groups']
	lb     = []
	im     = []
	for g in gids:
		try:
			isInside = is_group_in_geo(prms, lbData[g])
			if isInside:
				lb.append(lbData[g])
		except:
			pdb.set_trace()
	return lb

##
#Get all the raw labels
def get_raw_labels_all(prms, setName='train'):
	keys = get_folder_keys(prms)
	lb   = []
	for k in keys:
		lb = lb + get_raw_labels(prms, k, setName=setName)
	return lb
	
##
#Process the labels according to prms
def get_labels(prms, setName='train'):
	#The main quantity that requires randomization is patch matching
	#So we will base this code around that. 
	rawLb = get_raw_labels_all(prms, setName=setName)
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
				prob = randState.rand()
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
							quat = ru.euler2quat(z2-z1, y2-y1, x2-x1)
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

##
# Convert a prefix and folder into the image name
def prefix2imname(prms, prefixes):
	fList   = get_folder_list(prms)
	for ff in fList.keys():
		drName    = fList[ff].split('/')[-1]
		fList[ff] = drName
	print fList
	imNames = []
	for pf in prefixes:
		f1, p1, f2, p2 = pf
		if f2 is not None:
			imNames.append([osp.join(fList[f1], p1+'.jpg'), osp.join(fList[f2], p2 +'.jpg')])
		else:
			imNames.append([osp.join(fList[f1], p1+'.jpg'), None])
	return imNames

##
#Make the window files
def make_window_file(prms, setNames=['test', 'train']):
	if len(prms.labelNames)==1 and prms.labelNames[0] == 'nrml':
		numImPerExample = 1
	else:
		numImPerExample = 2	

	#Assuming the size of images
	h, w, ch = 640, 640, 3
	hCenter, wCenter = int(h/2), int(w/2)
	cr = int(prms.crpSz/2)
	minH = max(0, hCenter - cr)
	maxH = min(h, hCenter + cr)
	minW = max(0, wCenter - cr)
	maxW = min(w, wCenter + cr)  

	for s in setNames:
		#Get the im-label data
		lb, prefix = get_labels(prms, s)
		imNames1 = prefix2imname(prms, prefix)
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




#Polygon should be of type mplPath	
def show_images(prms, folderId):
	imNames, _ = folderid_to_im_label_files(prms, folderId)	
	plt.ion()
	for imn in imNames:
		im = plt.imread(imn)
		plt.imshow(im)
		inp = raw_input('Press a key to continue')
		if inp=='q':
			return

#Show the groups of images that have the same target point
def show_image_groups(prms, folderId):
	grps = get_target_groups(prms, folderId)
	imNames, lbNames = folderid_to_im_label_files(prms, folderId)
	plt.ion()
	plt.figure()
	for ig, g in enumerate(grps[0:-1]):
		st = g
		en = grps[ig+1]
		print (st,en)
		count = 0
		axl = []
		pltCount = 0
		for i in range(st,en):
			im = plt.imread(imNames[i])
			lb = parse_label_file(lbNames[i])
			if lb.align is not None:
				isAlgn = True
				loc = (lb.align.loc[0], lb.align.loc[1])
				#loc = (lb.align.loc[1], lb.align.loc[0])
			else:
				isAlgn = False
				print ('Align info not found')
				rows, cols, _ = im.shape
				loc = (int(rows/2.0), int(cols/2.0))
			if count < 9:
				ax = plt.subplot(3,3,count+1)
				if isAlgn:
					im = mydisp.box_on_im(im, loc, 27)
				else:
					im = mydisp.box_on_im(im, loc, 27, 'b')
				print im.shape
				ax.imshow(im)
				ax.set_title(('cm: (%.4f, %.4f, %.4f)'
											+ '\n dist: %.4f, head: %.3f, pitch: %.3f, yaw: %3f')\
										% (tuple(lb.pts.camera + [lb.dist] + lb.rots))) 	
				plt.draw()
				axl.append(ax)
				pltCount += 1
			count += 1
		inp = raw_input('Press a key to continue')
		if inp=='q':
			return
		for c in range(pltCount):
			axl[c].cla()
