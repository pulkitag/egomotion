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
import street_utils as su
import street_params as sp
import scipy.misc as scm
from multiprocessing import Pool, Manager, Queue, Process
import time

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

##
#Untar the files and then delete them. 
def untar_and_del(prms):
	fNames = get_tar_files(prms)	
	suffix = []
	for f in fNames:
		suffix.append(osp.split(f)[1])
	fNames = [osp.join(prms.paths.tar.dr, s) for s in suffix]
	for f in fNames:
		if not osp.exists(f):
			continue
		subprocess.check_call(['tar -xf %s -C %s' % (f, prms.paths.raw.dr)],shell=True) 
		subprocess.check_call(['rm %s' % f],shell=True) 
	return fNames	

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
# Save the keys of only the folders for which alignment data is available. 
def save_aligned_keys(prms):
	keys, names = su.get_folder_keys_all(prms)
	with open(prms.paths.proc.folders.aKey,'w') as f:
		for k,n in zip(keys,names):	
			if 'Aligned' in n:
				f.write('%s\n' % k)

##
#Save the non_aligned keys
def save_non_aligned_keys(prms):
	keys, names = su.get_folder_keys_all(prms)
	with open(prms.paths.proc.folders.naKey,'w') as f:
		for (k,n) in zip(keys, names):
			_,suffix  = osp.split(n)
			isAligned = False
			for n2 in names:
				if 'Aligned' in n2 and suffix in n2:
					isAligned=True
			if not isAligned:
				print n

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
# Save prefixes for each folder
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
		
##
# Store for each folder the number of prefixes
# and number of groups. 	
def save_counts(prms):
	keys,_ = su.get_folder_keys_all(prms)	
	prefixCount = edict()
	groupCount  = edict()
	for k in keys:
		print(k)
		prefix = su.get_prefixes(prms, k)
		prefixCount[k] = len(prefix)
		grps = su.get_target_groups(prms, k)
		groupCount[k] = len(grps)
	pickle.dump({'prefixCount': prefixCount, 'groupCount': groupCount},
						 open(prms.paths.proc.countFile, 'w'))

##
# Save the groups
def save_groups(prms, isAlignedOnly=True):
	grpKeyStr = prms.paths.grpKeyStr
	if isAlignedOnly:
		keys   = su.get_folder_keys_aligned(prms)	
	else:
		keys,_ = su.get_folder_keys_all(prms)	
	for k in keys:
		imNames, lbNames, prefixes = su.folderid_to_im_label_files(prms, k, opPrefix=True)
		print(k)
		#Determine groups
		grps = su.get_target_groups(prms, k)
		#Read the labels of each group and save them
		grpLabels = edict()
		for ig, g in enumerate(grps[0:-1]):
			st = g
			en = grps[ig+1]
			grpKey = grpKeyStr % ig	
			grpLabels[grpKey]      = edict()
			grpLabels[grpKey].num  = en - st
			grpLabels[grpKey].prefix   = []
			grpLabels[grpKey].data     = []
			grpLabels[grpKey].folderId = k
			for i in range(st,en):
				grpLabels[grpKey].data.append(su.parse_label_file(lbNames[i]))
				grpLabels[grpKey].prefix.append(prefixes[i])
		pickle.dump({'groups': grpLabels}, 
							open(prms.paths.label.grps % k, 'w'))	

##
#Save geo localized groups
def save_geo_groups(prms):
	keys = su.get_folder_keys(prms)
	for k in keys:
		print (k)
		grpFile  = prms.paths.label.grps % k
		grpDat   = pickle.load(open(grpFile, 'r'))
		grpDat   = grpDat['groups']
		geoGrp   = edict()
		geoKeys  = []
		for gKey, gDat in grpDat.iteritems(): 
			isInside = su.is_group_in_geo(prms, gDat)
			if isInside:
				geoGrp[gKey] = gDat
				geoKeys.append(gKey)
		outName = prms.paths.grp.geoFile % k
		print ('Saving to %s' % outName) 
		pickle.dump({'groups': geoGrp, 'groupIds': geoKeys}, open(outName,'w'))

##
#Get the prefixes for a particular geo group
def get_prefixes_geo(prms, folderId):
	data = pickle.load(open(prms.paths.grp.geoFile % folderId))
	grps = data['groups']
	pref = []
	for _,g in grps.iteritems():
		for n in range(g.num):
			pref.append(g.prefix[n].strip())
	return pref

##
#Get all the geo prefixes
def get_prefixes_geo_all(prms):
	keys = su.get_folder_keys(prms)
	pref = edict()
	for k in keys:
		pref[k] = get_prefixes_geo(prms, k)
	return pref

##
#Helper for save_resize_images_geo
def _write_im(prms, readList, outNames):
	if prms.isAligned:
		rootDir = osp.join(cfg.STREETVIEW_DATA_MAIN, 
							'pulkitag/data_sets/streetview/raw/ssd105/Amir/WashingtonAligned/')
	else:
		raise Exception('rootDir is not defined')
	rdNames = su.prefix2imname(prms, readList)
	for r in range(len(rdNames)):
		#print (rdNames[r][0], outNames[r])
		im       = scm.imread(osp.join(rootDir, rdNames[r][0]))
		#Resize
		h, w, ch = im.shape
		hSt = max(0,int(h/2 - prms.rawImSz/2))
		wSt = max(0,int(w/2 - prms.rawImSz/2))
		hEn = min(h, int(hSt + prms.rawImSz))
		wEn = min(w, int(wSt + prms.rawImSz))
		im =  im[hSt:hEn, wSt:wEn, :] 
		#Save the image
		scm.imsave(outNames[r], im)

##
#Save cropped images
def save_cropped_images_geo(prms):
	pref    = get_prefixes_geo_all(prms)
	imKeys  = edict()
	imCount = 0 		
	l1Count, l2Count = 0,0
	l1Max = 1e+6
	l2Max = 1e+3
	readList  = []
	outNames  = []
	for k in pref.keys():
		imKeys[k]= edict()
		print (k, imCount)
		for i in range(len(pref[k])):
			imNum  = imCount % 1000
			imName = 'l1-%d/l2-%d/im%04d.jpg' % (l1Count, l2Count, imNum)
			imKeys[k][pref[k][i]] = imName
			imName = osp.join(prms.paths.proc.im.dr, imName)
			imDir,_ = osp.split(imName)
			sp._mkdir(imDir)
			readList.append((k, pref[k][i], None, None))
			outNames.append(imName)
			
			#Increment the counters
			imCount = imCount + 1
			l1Count = int(np.floor(imCount/l1Max))
			l2Count = int(np.floor((imCount % l1Max)/l2Max))
			
			#Write the images if needed
			if (imCount >= l2Max and (imCount % l2Max)==0):
				print (imCount)
				_write_im(prms, readList, outNames)	
				readList, outNames = [], []
	_write_im(prms, readList, outNames)
	pickle.dump({'imKeys':imKeys}, open(prms.paths.proc.im.keyFile,'w'))	

##
#Filter groups by distance
def filter_groups_by_dist(groups, seedGroups, minDist):
	'''
		groups     is a dict
		seedGroups is a dict/list
	'''
	grpKeys = []
	if type(seedGroups) is list:
		itr = enumerate(seedGroups)
	else:
		itr = seedGroups.iteritems()
	for (i,k) in enumerate(groups.keys()):
		#print (i)
		g      = groups[k]
		#Find min distance from all the seed groups
		sgDist = np.inf
		for _,sg in itr:
			dist = su.get_distance_groups(g, sg)
			if dist < sgDist:
				sgDist = dist
		if sgDist > minDist:
			grpKeys.append(k)
	return [sgDist]
	#return grpKeys
		
def _filter_groups_by_dist(args):
	return filter_groups_by_dist(*args)

##
#Filter groups by dist parallel
def p_filter_groups_by_dist(prms):
	pool = Pool(processes=32)
	seedGrps = su.get_groups(prms, '0052', setName=None)
	grps     = su.get_groups(prms, '0048', setName=None)
	print (len(seedGrps), len(grps))
	t1 = time.time()
	inArgs = []
	for gk in grps.keys():
		inArgs.append(({'%s'%gk:grps[gk]}, seedGrps, prms.splits.dist))
	res    = pool.map_async(_filter_groups_by_dist, inArgs) 
	trKeys = res.get()
	t2     = time.time()
	print ("Time: %f" % (t2-t1))
	trKeys = [tk[0] for tk in trKeys if not(tk==[])]  
	del pool
	return trKeys


##
#Save the splits data
def save_train_test_splits(prms, isForceWrite=False):
	if prms.splits.dist is None:
		save_train_test_splits_old(prms, isForceWrite=isForceWrite)
		return None

	keys = su.get_folder_keys(prms)
	keys = ['0052']
	pool = Pool(processes=32)
	for k in keys:
		fName = prms.paths.proc.splitsFile % k
		if os.path.exists(fName) and isForceWrite:
			print('%s already exists' % fName)
			#inp = raw_input('Are you sure you want to overwrite')
			#if not (inp == 'y'):
			#	return
		if osp.exists(fName) and not isForceWrite:
			print ('%s exists, skipping' % fName)	
			continue

		#Form the random seed
		randSeed  = prms.splits.randSeed + 2 * int(k)	
		randState = np.random.RandomState(randSeed) 
	
		#Read the groups	
		grps    = su.get_groups(prms, k, setName=None)
		grpKeys = grps.keys()
		N    = len(grpKeys)
		print('Folder: %s, num groups: %d' % (k,N))
		if N == 0:
			trKeys, valKeys, teKeys = [], [], []
		else:
			#Chose the test groups
			teN  = int((prms.splits.tePct/100.0) * N)	
			perm = randState.permutation(N)
			tePerm = perm[0:teN]
		
			teKeys = [grpKeys[t]for t in tePerm]
			teGrps = [grps[k] for k in teKeys]
			pList  = []
			t1 = time.time()
			inArgs = []
			for gk in grpKeys:
				inArgs.append(({'%s'%gk:grps[gk]}, teGrps, prms.splits.dist))
			res    = pool.map_async(_filter_groups_by_dist, inArgs) 
			trKeys = res.get()
			t2     = time.time()
			print ("Time: %f" % (t2-t1))
			trKeys = [tk[0] for tk in trKeys if not(tk==[])]  
			return trKeys
			for tk in teKeys:
				assert tk not in trKeys, 'THERE IS SOMETHING WRONG'
			print ('Num Test: %d, Num Train: %d' % (len(teKeys), len(trKeys)))
			valKeys = [k for k in grpKeys if (k not in teKeys) and (k not in trKeys)]

		#Save the splits
		splits = edict()
		splits.train = trKeys	
		splits.test  = teKeys
		splits.val   = valKeys
		#Save the data		
		pickle.dump({'splits': splits}, open(fName, 'w'))


##
#The old hacky way of generating train-test splits
def save_train_test_splits_old(prms, isForceWrite=False):
	keys = su.get_folder_keys(prms)
	for k in keys:
		fName = prms.paths.proc.splitsFile % k
		if os.path.exists(fName) and isForceWrite:
			print('%s already exists' % fName)
			#inp = raw_input('Are you sure you want to overwrite')
			#if not (inp == 'y'):
			#	return
		if osp.exists(fName) and not isForceWrite:
			print ('%s exists, skipping' % fName)	
			continue

		#Form the random seed
		randSeed  = prms.splits.randSeed + 2 * int(k)	
		randState = np.random.RandomState(randSeed) 
		#Read the groups of the fodler
		grps = ['%07d' % ig for (ig,g) in enumerate(su.get_target_groups(prms, k)[0:-1])]
		N    = len(grps)
		print('Folder: %s, num groups: %d' % (k,N))
		teN  = int((prms.splits.tePct/100.0) * N)	
		perm = randState.permutation(N)
		tePerm = perm[0:teN]
		#Form an extended testing set to exclude the neighbors
		valPerm = []
		print ('Extending test set for buffering against closeness to train set')
		for t in tePerm:
			st = max(0, t - prms.splits.teGap)
			en = min(len(grps), t + prms.splits.teGap+1)
			valPerm = valPerm + [v for v in range(st, en)]
		print ('Form the train set')
		#Form the training set
		trPerm = [t for t in perm if t not in valPerm]
		splits = edict()
		splits.train = [grps[g] for g in trPerm]		
		splits.test  = [grps[g] for g in tePerm]
		splits.val   = [grps[g] for g in valPerm if g not in tePerm]
		#Save the data		
		pickle.dump({'splits': splits}, open(fName, 'w'))

	
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


