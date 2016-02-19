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
import math

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
#Get the displacement vector from coordinates 
#expresses as latitude, longitude, height
def get_displacement_vector(pt1, pt2):
	'''
		pt1: lat1, long1, height1
		pt2: same format as pt1
	'''	
	lat1, long1, h1 = pt1
	lat2, long2, h2 = pt2
	lat1, long1 = lat1*np.pi/180., long1*np.pi/180.
	lat2, long2 = lat2*np.pi/180., long2*np.pi/180.
	R = 6371.0088 * 1000 #Earth's average radius in m
	y = R * (lat2 - lat1)
	x = R * (long2 - long1) * math.acos((lat1 + lat2)/2.0)
	z = h2 - h1
	return x, y, z

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



