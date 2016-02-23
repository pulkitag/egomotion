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
import my_exp_pose_v2 as mepo2

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
#Test the displacement computed using get_displacement_vector
def _test_get_displacement_vector():
	prms,_ = mepo2.smallnetv5_fc5_pose_euler_5dof_crp192_rawImSz256_lossl1()
	grps = get_groups(prms, '0001')
	errs = []
	gAll, aAll = [], []
	for g in grps:
		N    = g.num
		perm = np.random.permutation(N)
		if N < 2:
			continue	
		p1, p2 = perm[0], perm[1]
		pt1   = g.data[p1].pts.camera[0:3]
		pt2   = g.data[p2].pts.camera[0:3]
		gDist = geodist(pt1[0:2], pt2[0:2]).meters
		x, y, z = get_displacement_vector(pt1, pt2)
		aDist   = np.sqrt(x*x + y*y)
		errs.append(gDist - aDist)
		gAll.append(gDist)
		aAll.append(aDist)
	errs, gAll, aAll = np.array(errs), np.array(gAll), np.array(aAll)
	rErr = np.abs(errs)/np.minimum(np.abs(gAll), np.abs(aAll))
	print 'Mean relative Error: %f, sd relative error: %f' % (np.mean(rErr), np.std(rErr))
	return errs, rErr, gAll, aAll

##
#Get the prefixes for a specific folderId
def get_prefixes(prms, folderId):
	with open(prms.paths.proc.folders.pre % folderId,'r') as f:
		prefixes = [p.strip() for p in f.readlines()]
	return prefixes

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
#Get the train and test splits
def get_train_test_splits(prms, folderId):
	fName  = prms.paths.proc.splitsFile % folderId
	splits = edict(pickle.load(open(fName,'r')))
	return splits.splits

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
			
