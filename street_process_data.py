from easydict import EasyDict as edict
import pickle
import os
from os import path as osp
import copy
import other_utils as ou
import socket
import numpy as np
import pdb
import scipy.misc as scm
from geopy.distance import vincenty as geodist
from multiprocessing import Pool
import math
from matplotlib import cm as cmap
import matplotlib.pyplot as plt
import street_config as cfg
import street_exp_v2 as sev2
import subprocess
import scp_utils as scput

def get_config_paths():
	return cfg.pths

def save_cropped_image_unaligned(inNames, outNames, rawImSz=256, 
											isForceWrite=False):
	N = len(inNames)
	assert N == len(outNames)
	for i in range(N):
		if osp.exists(outNames[i]) and not isForceWrite:
			continue
		im         = scm.imread(inNames[i])
		h, w, ch = im.shape
		cy, cx = h/2, w/2
		#Crop
		hSt = min(h, max(0,int(cy - rawImSz/2)))
		wSt = min(w, max(0,int(cx - rawImSz/2)))
		hEn = min(h, int(hSt + rawImSz))
		wEn = min(w, int(wSt + rawImSz))
		imSave = np.zeros((rawImSz, rawImSz,3)).astype(np.uint8)
		hL, wL  = hEn - hSt, wEn - wSt
		#print hSt, hEn, wSt, wEn
		imSave[0:hL, 0:wL,:] =  im[hSt:hEn, wSt:wEn, :] 
		#Save the image
		dirName, _ = osp.split(outNames[i])
		ou.mkdir(dirName)
		scm.imsave(outNames[i], imSave)


def save_cropped_image_aligned(inNames, outNames, imCenter,
            rawImSz=256, isForceWrite=False):
	N = len(inNames)
	assert N == len(outNames)
	for i in range(N):
		if osp.exists(outNames[i]) and not isForceWrite:
			continue
		im         = scm.imread(inNames[i])
		h, w, ch = im.shape
		cy, cx = imCenter
		#Crop
		hSt = min(h, max(0,int(cy - rawImSz/2)))
		wSt = min(w, max(0,int(cx - rawImSz/2)))
		hEn = min(h, int(hSt + rawImSz))
		wEn = min(w, int(wSt + rawImSz))
		imSave = np.zeros((rawImSz, rawImSz,3)).astype(np.uint8)
		hL, wL  = hEn - hSt, wEn - wSt
		#print hSt, hEn, wSt, wEn
		imSave[0:hL, 0:wL,:] =  im[hSt:hEn, wSt:wEn, :] 
		#Save the image
		dirName, _ = osp.split(outNames[i])
		ou.mkdir(dirName)
		scm.imsave(outNames[i], imSave)


	
def get_distance_between_groups(grp1, grp2):
	lb1, lb2 = grp1.data[0], grp2.data[0]
	tPt1  = lb1.label.pts.target
	tPt2  = lb2.label.pts.target
	tDist = geodist(tPt1, tPt2).meters
	return tDist
	
def prune_groups(grps, gList1, gList2, minDist):
	'''
		grps  : a dict containing the group data
		gList1: a list of group keys
		gList2: a list of group keys
		Remove all keys from gList2 that are less than minDist
		from groups in gList1
	'''
	badList  = []
	gListTmp = copy.deepcopy(gList2) 
	for i1, gk1 in enumerate(gList1):
		if np.mod(i1,1000)==1:
			print i1, len(badList)
		for gk2 in gListTmp:
			tDist = get_distance_between_groups(grps[gk1].grp, grps[gk2].grp)
			if tDist < minDist:
				badList.append(gk2)
		gListTmp = [g for g in gListTmp if g not in badList]
	return gListTmp

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
#Geo distance calculations
class GeoCoordinate(object):
	def __init__(self, latitude, longitude, z=0, isRadian=False):
		#convert radians
		if not isRadian:
			self.lat_  = np.pi * latitude/180.
			self.long_ = np.pi * longitude/180.
		self.z_    = z
		#in meters
		self.earthRadius = 6371.0088 * 1000

	@classmethod
	def from_point(cls, pt):
		la, lo, z = pt
		return cls(la, lo, z)

	def get_xyz(self):
		#Approximate xyz coordinates in meters
		y = self.earthRadius * self.lat_
		x = self.earthRadius * self.long_ * math.acos(self.lat_)
		return x, y, self.z_

	def get_xy(self):
		x, y, z = self.get_xyz()
		return x, y

	def get_displacement_vector(self, pt2):
		'''
			pt2: the point to which displacement vector
					 is to be planned
		'''	
		y = R * (pt2.lat_ - self.lat_)
		x = R * (pt2.long_ - self.long_) * math.acos((self.lat_ + pt2.lat_)/2.0)
		z = pt2.z_ - pt1.z_
		return x, y, z



#Mantains a list of folders that have been processed
#and provides a id for folder path
class FolderStore(object):
	'''
		attributes
		file_   : File that stores the list of folders
		folders : dict storing list of folders
		keyStr  : base string using which keys are generated
		count   : number of folders that are here in the store
	'''
	def __init__(self, expStr='streetview'):
		pths  = get_config_paths()
		fPath = osp.join(pths.cwd, 'exp-data', 'folder')
		ou.mkdir(fPath)
		self.file_   = osp.join(fPath, '%s-folderlist.pkl' % expStr)
		self.folders = edict() 
		self.keyStr  = '%04d'
		self.count   = 0
		if osp.exists(self.file_):
			self._load()
		
	def _load(self):
		self.folders = pickle.load(open(self.file_, 'r'))
		self.count   = len(self.folders.keys())
	
	def is_present(self, fName):
		for k, v in self.folders.iteritems():
			if v == fName:
				return True
		return False
			
	def _append(self, fName):
		isPresent = self.is_present(fName)
		if isPresent:
			print ('folder %s already exists, cannot append to store' % fName)
			return
		self.folders[self.keyStr % self.count] = fName
		self.count += 1
		pickle.dump(self.folders, open(self.file_, 'w'))
		self._load()	
	
	def get_id(self, fName):
		if not self.is_present(fName):
			self._append(fName)
		for k, v in self.folders.iteritems():
			if v == fName:
				return k

	def get_list(self):
		return copy.deepcopy(self.folders)

	#Return the name of the folder form the id
	def get_folder_name(self, fid):
		assert fid in self.folders.keys(), '%s folder id not found' % fid
		return self.folders[fid]


class StreetLabel(object):
	def __init__(self):
		self.label = edict()

	def get(self):
		return copy.deepcopy(self.label)

	@classmethod
	def from_file(cls, fName):
		self = cls()
		with open(fName, 'r') as f:
			data = f.readlines()
			if len(data)==0:
				return None
			dl   = data[0].strip().split()
			assert dl[0] == 'd'
			self.label.ids    = edict()
			self.label.ids.ds = int(dl[1]) #DataSet id
			self.label.ids.tg = int(dl[2]) #Target id
			self.label.ids.im = int(dl[3]) #Image id
			self.label.ids.sv = int(dl[4]) #Street view id
			self.label.pts    = edict()
			self.label.pts.target = [float(n) for n in dl[5:8]] #Target point
			self.label.nrml   = [float(n) for n in dl[8:11]]
			#heading, pitch , roll
			#to compute the rotation matrix - use xyz' (i.e. first apply pitch, then 
			#heading and then the rotation about the camera in the coordinate system
			#of the camera - this is equivalent to zxy format.  
			self.label.pts.camera = [float(n) for n in dl[11:14]] #street view point not needed
			self.label.dist   = float(dl[14])
			self.label.rots   = [float(n) for n in dl[15:18]]
			assert self.label.rots[2] == 0, 'Something is wrong %s' % fName	
			self.label.align = None
			if len(data) == 2:
				al = data[1].strip().split()[1:]
				self.label.align = edict()
				#Corrected patch center
				self.label.align.loc	 = [float(n) for n in al[0:2]]
				#Warp matrix	
				self.label.align.warp = np.array([float(n) for n in al[2:11]])
		return self

	@classmethod
	def from_dict(cls, label):
		self.label = copy.deepcopy(label)	
		
	def is_aligned(self):
		if self.label.align is not None:
			return True
		else:
			return False
	
	def get_rot_euler(self, isRadian=True):
		pass

	def get_normals(self):
		pass

	def get_rot_mat(self):
		pass


class StreetGroup(object):
	def __init__(self, grp=edict()):
		self.grp = copy.deepcopy(grp)

	@classmethod
	def from_raw(cls, folderId, gId, prefix, lbNames, crpImNames):
		grp = edict()
		grp.data     = []
		grp.folderId = folderId
		grp.gid      = gId
		grp.prefix     = []
		grp.crpImNames = []
		grp.num        = 0
		for i,p in enumerate(prefix):
			bsName = osp.basename(lbNames[i]).split('.')[0]
			bsGid  = bsName.split('_')[3]
			assert bsName == p, 'bsName:%s, p: %s'% (bsName,p)
			assert bsGid  == gId, 'Group name mismatch: %s, %s' % (bsGid, gid)
			slb = StreetLabel.from_file(lbNames[i])
			if slb is not None:
				grp.data.append(slb)
				grp.prefix.append(p)
				grp.crpImNames.append(crpImNames[i])
				grp.num += 1
		return cls(grp)

	def distance_from_other(self, grp2):
		return get_distance_between_groups(self.grp, grp2.grp)

	#Form a subset using idx
	def subset(self, idx):
		'''
			idx: list of indices
		'''
		grp = edict()
		grp.folderId = self.grp.folderId
		grp.gid      = self.grp.gid
		grp.num      = len(idx)
		grp.crpImNames = []
		grp.prefix     = []
		grp.data       = []
		for i in idx:
			grp.crpImNames.append(self.grp.crpImNames[i])
			grp.prefix.append(self.grp.prefix[i])
			grp.data.append(copy.deepcopy(self.grp.data[i]))
		return StreetGroup(grp)
			
	#Get a subset of only aligned data
	def subset_aligned(self):
		idx = []
		for i,lb in enumerate(self.grp.data):
			if lb.is_aligned():
				idx.append(i)
		if len(idx) == 0:
			return None
		else:
			return self.subset(idx)	
	
	def as_dict(self):
		grpDict = edict()
		for k in self.grp.keys():
			if k == 'data':
				continue
			grpDict[k] = self.grp[k]	
		grpDict['data'] = []
		for d in self.grp.data:
			grpDict['data'].append(d.label)
		return grpDict


class StreetGroupList(object):
	def __init__(self):
		self.grps  = edict()
		self.gKeys = []

	@classmethod
	def from_dict(cls, grps, gKeys):
		self = cls()
		#self.grps  = copy.deepcopy(grps)
		#self.gKeys = copy.deepcopy(gKeys)
		self.grps  = grps
		self.gKeys = gKeys
		return self

	#get the x,y point coordinates all the targets
	def get_target_xy(self):
		xy = np.zeros((len(self.gKeys),2))
		for i, gk in enumerate(self.gKeys):	
			tPt = GeoCoordinate.from_point(self.grps[gk].grp.data[0].label.pts.target)
			xy[i,:] = tPt.get_xy()
		return xy	

	#plot the x,y points
	def plot_xy(self):
		xy = self.get_target_xy()
		xMin, yMin = np.min(xy,0)
		xMax, yMax = np.max(xy,0)
		xy  = xy - np.array([xMin, yMin])
		xy  = xy / np.array([xMax - xMin, yMax - yMin])
		perm       = np.random.permutation(xy.shape[0])
		perm       = perm[0:int(0.2*perm.shape[0])]
		plt.ion()
		fig        = plt.figure()
		ax         = fig.add_subplot(111)
		for i in perm:
			ax.plot(xy[i,0], xy[i,1], '.r')
		plt.draw()
		plt.show()	
	
	#grid x,y locations
	def grid_xy(self):
		xy = self.get_target_xy()
		xMin, yMin = np.min(xy, 0)
		xMax, yMax = np.max(xy, 0)
		xLen, yLen = np.ceil(xMax - xMin), np.ceil(yMax - yMin)
		numX, numY = np.ceil(xLen/100), np.ceil(yLen/100)
		xBins = np.linspace(xMin, xMax+1, numX)
		yBins = np.linspace(yMin, yMax+1, numY)
		#Initialize the lists
		self.binList = []
		for x in xBins:
			yList = []
			for y in yBins:	
				yList.append([])
			self.binList.append(yList)
		#Find the bin index of all groups
		for i, gk in enumerate(self.gKeys):
			xIdx = find_bin_index(xBins, xy[i,0])
			yIdx = find_bin_index(yBins, xy[i,1])
			self.binList[xIdx][yIdx].append(gk)
		self.grid_count()

	#get groups present in a certain grid location
	def get_groups_in_gridxy(self, xy):
		x, y = xy
		return copy.deepcopy(self.binList[x][y])

	#Count of number of groups in this bin
	def grid_count(self):
		nX, nY = len(self.binList), len(self.binList[0])
		self.gridBinCount = np.zeros((nY, nX))
		for x in range(nX):
			for y in range(nY):
				self.gridBinCount[y, x] = len(self.binList[x][y])

	def plot_grid(self, ax=None):
		if ax is None:
			fig = plt.figure()
			ax  = fig.add_subplot(111)
		mx = np.max(self.gridBinCount)
		im = self.gridBinCount/np.max(self.gridBinCount)
		cax = ax.imshow(im, interpolation="nearest", cmap='jet')
		cbar = fig.colorbar(cax, ticks=np.linspace(0,1,11))
		cbar.set_ticklabels(mx * np.linspace(0,1,11))
		plt.draw()
		print ('Number of groups: %d' % np.sum(self.gridBinCount))

	def _update_grid_count(self, cx, cy, mxCount):
		#print (cx, cy, self._gridCount_)
		self._gridCount_ += len(self.binList[cx][cy])
		breakFlag   = False
		if self._gridCount_ >= mxCount:
			breakFlag = True
		return breakFlag

	#Divide the group into two halves such that
	def divide_group_counts(self, mxCount):
		'''
			This has been engineered to the use case of splitting the data
			into train/val splits. Therefore we leave out all elements in 
			the first/last row/col - to avoid any conflicts with elements in 
      other groups.
			The groups are chosen in row-major fashion starting at (1,1) 
		'''
		nX, nY = len(self.binList), len(self.binList[0])
		self._gridCount_ = 0
		#Select the groups
		gBins      = []
		mxX = 1
		for cy in range(1, nY-1):
			for cx in range(1, nX-1):
				gBins.append((cx, cy))
				breakFlag = self._update_grid_count(cx, cy, mxCount)
				if cx > mxX:
					mxX = cx
				if breakFlag:
					break
			if breakFlag:
				break
		#Select a border of groups so that groups in two
		#sets are seperated by 100m. 
		bBins = []
		for x in range(cx+1, min(mxX + 2,nX-1)):
			bBins.append((x, cy))
		for x in range(0, min(nX-1,cx+2)):
			bBins.append((x, cy+1))
		#Include the top and bottom border
		for x in range(nX):
			bBins.append((x, 0))
			bBins.append((x, nY-1))
		#Include the left and right border
		for y in range(nY):
			bBins.append((0,y))
			bBins.append((nX-1,y))
		bBins = list(set(bBins))	
		return gBins, bBins

	def get_split_groups(self, splitCount):
		self.grid_xy()
		gBins, ignoreBins = self.divide_group_counts(splitCount)
		keys1, keys2 = [], []
		for g in gBins:
			keys1 = keys1 + self.get_groups_in_gridxy(g)
		igKeys = []
		for g in ignoreBins:
			igKeys = igKeys + self.get_groups_in_gridxy(g)
		igKeys = igKeys + keys1
		keys2  = [gk for gk in self.gKeys if gk not in igKeys]
		return keys1, keys2
		

	def vis_divided_grid(self, gBins, bBins):
		'''
			gBins: grid bins
			bBins: chosen bins
		'''
		nX, nY = len(self.binList), len(self.binList[0])
		plt.ion()
		fig    = plt.figure()
		ax1     = fig.add_subplot(221)
		ax2     = fig.add_subplot(222)
		ax3     = fig.add_subplot(223)
		im = np.zeros((nY, nX, 3))
		imBoth = np.zeros((nY, nX, 3))
		for i, g in enumerate(gBins):
			x, y = g	
			im[y,x,:] = cmap.jet(int(255*float(i)/len(gBins)))[0:3]
			imBoth[y,x,:] = cmap.jet(int(255*float(i)/len(gBins)))[0:3]
		ax1.imshow(im, interpolation="nearest")
		im2 = np.zeros((nY, nX, 3))
		for i, g in enumerate(bBins):
			x, y = g	
			im2[y,x,:] = 1.0, 1.0, 1.0
			imBoth[y,x,:] = 1.0, 1.0, 1.0
		ax2.imshow(im2, interpolation="nearest")
		ax3.imshow(imBoth, interpolation="nearest")
		plt.show()
		plt.draw()

def _debug_group_list_div(grps, grpList, mxCount=5000):
	sgList = StreetGroupList.from_dict(grps, grpList)
	sgList.grid_xy()
	div, bdiv = sgList.divide_group_counts(mxCount)
	sgList.vis_divided_grid(div, bdiv)

class StreetFolder(object):
	'''
		prefix: prefix.jpg, prefix.txt define the corresponding image and lbl file
		target_group: group of images looking at the same target point. 
	'''
	def __init__(self, name, splitPrms=None, isAlign=False):
		'''
			name:  relative path where the raw data in the folder is present
			svPth: the path where the processed data from the folder should be saved
		'''
		#folder id
		fStore        = FolderStore('streetview')
		self.id_      = fStore.get_id(name)
		self.name_    = name
		self.isAlign_ = isAlign
		if splitPrms is None:
			self.splitPrms_ = sev2.get_trainval_split_prms() 
		#folder paths
		self._paths()
		#Prefixes
		self.prefixList_ = []
		self._read_prefixes_from_file()

	#
	def _paths(self):
		pths     = sev2.get_folder_paths(self.id_, self.splitPrms_, self.isAlign_)
		cPaths   = get_config_paths() 
		#raw data
		self.dirName_ = osp.join(cPaths.mainDataDr, self.name_)
		self.paths_   = pths
			
	#Save all prefixes in the folder
	def _save_prefixes(self):
		if not osp.exists(self.dirName_):
			print ('PREFIX CANNOT BE SAVED AS RAW DATA NOT PRESENT')
			return False
		allNames = os.listdir(self.dirName_)
		imNames   = sorted([f for f in allNames if '.jpg' in f], reverse=True)
		lbNames   = sorted([f for f in allNames if '.txt' in f], reverse=True)
		prefixStr = []
		for (i,imn) in enumerate(imNames):
			imn = imn[0:-4]
			if i>= len(lbNames):
				continue
			if imn in lbNames[i]:
				#Determine that the labelfile is not empty
				lbDat = StreetLabel.from_file(osp.join(self.dirName_, lbNames[i]))
				if lbDat is None:
					print ('%s has no label info' % lbNames[i])
					continue
				prefixStr = prefixStr + [imn]
		pickle.dump({'prefixStr': prefixStr}, open(self.paths_.prefix, 'w'))
		return True

	#Read the prefixes from saved file
	def _read_prefixes_from_file(self, forceCompute=False):
		''' Read all the prefixes
			forceCompute: True - recompute the prefixes
		'''
		if not forceCompute and osp.exists(self.paths_.prefix):
			dat = pickle.load(open(self.paths_.prefix,'r'))
			self.prefixList_ = dat['prefixStr']
			return	
		print ('Computing prefixes for folderid %s' % self.id_)	
		isSave = self._save_prefixes()	
		if not isSave:
			print ('PREFIX CANNOT BE READ')
			return False
		self._read_prefixes_from_file(forceCompute=False)


	#Get the prefixes
	def get_prefixes(self):
		return copy.deepcopy(self.prefixList_)

	#
	def get_im_label_files(self):
		imFiles = [osp.join(self.dirName_, '%s.jpg') % p for p in self.prefixList_]
		lbFiles = [osp.join(self.dirName_, '%s.txt') % p for p in self.prefixList_]
		return imFiles, lbFiles

	#Save group of images that look at the same target point. 	
	def _save_target_group_counts(self):
		if self.isAlign_:
			self._save_target_group_counts_aligned()
			return
		grps = {}
		prev     = None
		count    = 0
		tgtList  = []
		for (i,p) in enumerate(self.prefixList_):	
			_,_,_,grp = p.split('_')
			if i == 0:
				prev = grp
			if not(grp == prev):
				tgtList.append(prev)
				grps[prev]= count
				prev  = grp
				count = 0
			count += 1
		pickle.dump(grps, open(self.paths_.prePerGrp, 'w'))
		pickle.dump({'grpList': tgtList}, open(self.paths_.targetGrpList, 'w'))

	
	#get all the target group counts
	def get_num_prefix_per_target_group(self, forceCompute=False):
		if not forceCompute and osp.exists(self.paths_.prePerGrp):
			prePerGrp = pickle.load(open(self.paths_.prePerGrp, 'r'))
			return prePerGrp
		self._save_target_group_counts()
		return self.get_num_prefix_per_target_group()

	#ordered list of groups
	def get_target_group_list(self):
		if self.isAlign_:
			data = pickle.load(open(self.paths_.targetGrpListAlign, 'r'))
		else:
			data = pickle.load(open(self.paths_.targetGrpList, 'r'))
		return data['grpList']	
		
	#save the target group data
	def _save_target_groups(self, forceWrite=False):
		if self.isAlign_:
			self._save_target_groups_aligned(forceWrite=forceWrite)
			return
		if osp.exists(self.paths_.targetGrps) and not forceWrite:
			print ('Group file %s exists, NOT recomputing' % self.paths_.targetGrps)
			return
		print ('Computing group file %s' % self.paths_.targetGrps)
		imNames, lbNames = self.get_im_label_files()
		preCountPerGrp   = self.get_num_prefix_per_target_group()
		grps  = edict()
		grpKeys          = self.get_target_group_list()
		prevCount = 0
		for gk in grpKeys:
			numPrefix = preCountPerGrp[gk] 
			st, en    = prevCount, prevCount + numPrefix
			cropImNames = [self._idx2cropname(idx) for idx in range(st, en)]
			try:
				grps[gk] = StreetGroup.from_raw(self.id_, gk,
               self.prefixList_[st:en], lbNames[st:en], cropImNames)
			except:
				print ('#### ERROR Encountered #####')
				print (self.id_, gk, st, en)
				print (self.prefixList_[st:en])
				pdb.set_trace()
			prevCount += numPrefix
		print ('SAVING to %s' % self.paths_.targetGrps)
		pickle.dump({'groups': grps}, open(self.paths_.targetGrps, 'w'))	

	#Save the counts of aligned groups
	def _save_target_groups_aligned(self, forceWrite=False):
		assert self.isAlign_, 'Align_ is set to False'
		self.isAlign_ = False
		#Check is the original groups have been saved
		if not osp.exists(self.paths_.targetGrpList):
			self._save_target_group_counts()
		#Check if the aligned groups have already been saved
		if osp.exists(self.paths_.targetGrpsAlign) and not forceWrite:
			print ('Group file %s exists, NOT recomputing' % self.paths_.targetGrpsAlign)
			return
		print ('%s loading all groups' % self.id_)
		allGrpList = self.get_target_group_list()
		allGrps    = self.get_target_groups()
		alignGrpList = []
		alignGrps       = edict()
		alignPrefixList = []
		print ('%s filtering aligned groups' % self.id_)
		for gk in allGrpList:	
			alGrp = allGrps[gk].subset_aligned()
			if alGrp is not None:
				alignGrpList.append(gk)
				alignGrps[gk] = alGrp
				for p in alGrp.grp.prefix:
					alignPrefixList.append(p)
		print ('Folder: %s, number of groups: %d, number of align: %d' %
           (self.id_, len(allGrpList), len(alignGrpList)))
		pickle.dump({'grpList': alignGrpList}, open(self.paths_.targetGrpListAlign, 'w'))
		pickle.dump({'groups': alignGrps}, open(self.paths_.targetGrpsAlign, 'w'))	
		pickle.dump({'prefixStr': alignPrefixList}, open(self.paths_.prefixAlign, 'w'))	
		self.isAlign_ = True
		
	#get groups
	def get_target_groups(self):
		if self.isAlign_:
			dat = pickle.load(open(self.paths_.targetGrpsAlign, 'r'))
		else:
			dat = pickle.load(open(self.paths_.targetGrps, 'r'))
		return dat['groups']		

	#crop and save images - that makes it faster to read for training nets
	def save_cropped_images(self, imSz=256, isForceWrite=False):
		if self.isAlign_:
			self.save_cropped_images_aligned(imSz=imSz, isForceWrite=isForceWrite)
			return
		cropDirName = self.paths_.crpImPath % imSz
		ou.mkdir(cropDirName)
		for i, p in enumerate(self.prefixList_):
			if np.mod(i,1000)==1:
				print(i)
			inName    = osp.join(self.dirName_, p + '.jpg')
			outName   = osp.join(self.paths_.crpImPath % imSz, 
                 self._idx2cropname(i))
			#Save the image
			save_cropped_image_unaligned([inName], [outName],
            imSz, isForceWrite)

	def save_cropped_images_aligned(self, imSz=256, isForceWrite=False):
		assert self.isAlign_, 'self.Align_ must be True'
		print('%s, loading groups' % self.id_)
		grpList = self.get_target_group_list() 
		grps    = self.get_target_groups()
		count   = 0
		for gk in grpList:
			g = grps[gk]
			for n in range(g.grp.num):
				prf = g.grp.prefix[n]
				loc = g.grp.data[n].label.align.loc
				inName    = osp.join(self.dirName_, prf + '.jpg')
				outName   = osp.join(self.paths_.crpImPathAlign % imSz, 
									 self._idx2cropname(count))
				#Save the image
				save_cropped_image_aligned([inName], [outName], loc,
							imSz, isForceWrite)
				count += 1

	def _idx2cropname(self, idx):
		lNum  = int(idx/1000)
		imNum = np.mod(idx, 1000)
		name  = 'l-%d/im%04d.jpg' % (lNum, imNum) 
		return name

	def get_cropped_imname(self, prefix):
		idx = self.prefixList_.index(prefix)
		return self._idx2cropname(idx)

	#Split into train/test/val
	def split_trainval_sets(self, grps=None):
		sPrms = self.splitPrms_
		assert sPrms.minDist == 100, 'Modify StreetGroupList gridding to\
       work for other distance values'
		if grps is None:
			print ('Reading Groups')
			grps  = self.get_target_groups()
		gKeys = self.get_target_group_list()
		N     = len(gKeys)
		nTrn  = int(np.floor((sPrms.trnPct/100.0 * (N))))
		nVal  = int(np.floor((sPrms.valPct/100.0 * (N))))
		nTe   = int(np.floor((sPrms.tePct/100.0 * (N))))
		#Make a list of groups
		gList = StreetGroupList.from_dict(grps, gKeys)
		trnKeys, oKeys = gList.get_split_groups(nTrn)
		oL     = len(oKeys)
		tL     = len(trnKeys)
		assert N >= tL + oL
		print ('FolderId: %s, Num-Train: %d, Num-Others: %d, NumLost: %d' %
           (self.id_, tL, oL, N - (tL + oL)))
		valIdx  = int(oL * (float(sPrms.valPct)/(sPrms.valPct + sPrms.tePct))) 
		setKeys = edict()
		setKeys['train'] = [k for k in trnKeys if grps[k].grp.num > 1]
		setKeys['val']   = [k for k in oKeys[0:valIdx] if grps[k].grp.num > 1]
		setKeys['test']  = [k for k in oKeys[valIdx:] if grps[k].grp.num > 1] 
		print ('Num-Train: %d, Num-Val: %d, Num-Test: %d' % 
				 (len(setKeys['train']), len(setKeys['val']), len(setKeys['test'])))
		#Sanity checks
		assert len(set(setKeys['train']).intersection(set(setKeys['val']))) == 0
		assert len(set(setKeys['train']).intersection(set(setKeys['test']))) == 0
		#Save the split keys
		pickle.dump({'setKeys': setKeys, 'splitPrms': self.splitPrms_},
               open(self.paths_.trainvalSplitGrpKeys, 'w')) 
		for s in ['train', 'val', 'test']:
			sGroups = [grps[gk].as_dict() for gk in setKeys[s]]
			pickle.dump({'groups': sGroups}, open(self.paths_.grpSplits[s], 'w')) 	

	def tar_trainval_splits(self, forceWrite=False):
		drName  = self.paths_.deriv.grps
		trFile  = self.paths_.deriv.grpsTar
		if not osp.exists(trFile) or forceWrite:
			print ('Making %s' % trFile)
			subprocess.check_call(['tar -cf %s -C %s .' % (trFile, drName)],shell=True)
		else:
			print ('TAR file already exists')

	#untar the files in the correct directory
	def untar_trainval_splits(self):
		drName  = self.paths_.deriv.grps
		ou.mkdir(drName)
		trFile  = self.paths_.deriv.grpsTar
		subprocess.check_call(['tar -xf %s -C %s' % (trFile, drName)],shell=True)

	def tar_cropped_images(self, imSz=256, forceWrite=False):
		if self.isAlign_:
			drName  = self.paths_.crpImPathAlign % imSz
			trFile  = self.paths_.crpImPathAlignTar % imSz
		else:
			drName  = self.paths_.crpImPath % imSz
			trFile  = self.paths_.crpImPathTar % imSz
		dirName = osp.dirname(trFile)
		ou.mkdir(dirName)
		if not osp.exists(trFile) or forceWrite:
			print ('Making %s' % trFile)
			subprocess.check_call(['tar -cf %s -C %s .' % (trFile, drName)],shell=True)
		else:
			print ('TAR file already exists')

	#Transfer the trainval splits to a host
	def scp_trainval_splits(self, hostName):
		fPaths  = sev2.get_folder_paths(self.id_, self.splitPrms_,
              self.isAlign_, hostName)
		srcPath = self.paths_.deriv.grpsTar
		tgHost  = scput.get_hostaddr(hostName)
		tgPath  = tgHost + fPaths.deriv.grpsTar	
		print (tgPath) 
		subprocess.check_call(['rsync -ravz %s %s' % (srcPath, tgPath)],shell=True)

	#Fetch trainval splits from a host
	def fetch_scp_trainval_splits(self, hostName):
		fPaths  = sev2.get_folder_paths(self.id_, self.splitPrms_,
              self.isAlign_, hostName)
		tgPath  = self.paths_.deriv.grpsTar
		srcHost  = scput.get_hostaddr(hostName)
		srcPath  = srcHost + fPaths.deriv.grpsTar	
		print (srcPath) 
		dirName = osp.dirname(tgPath)
		ou.mkdir(dirName)
		#subprocess.check_call(['rsync -ravz %s %s' % (srcPath, tgPath)],shell=True)
		subprocess.check_call(['scp %s %s' % (srcPath, tgPath)],shell=True)

	#Transfer the cropped images to a host
	def fetch_scp_cropped_images(self, hostName, imSz=256):
		print ('I AM HERE')
		fPaths  = sev2.get_folder_paths(self.id_, self.splitPrms_,
              self.isAlign_, hostName)
		srcHost  = scput.get_hostaddr(hostName)
		if self.isAlign_:
			tgPath  = self.paths_.crpImPathAlignTar % imSz
			srcPath = srcHost + fPaths.crpImPathAlignTar % imSz
		else:
			tgPath  = self.paths_.crpImPathTar % imSz
			srcPath = srcHost + fPaths.crpImPathTar % imSz
		dirName = osp.dirname(tgPath)
		print (dirName)
		if not osp.exists(dirName):
			ou.mkdir(dirName)
		print (srcPath) 
		#subprocess.check_call(['rsync -ravz %s %s' % (srcPath, tgPath)],shell=True)
		subprocess.check_call(['scp %s %s' % (srcPath, tgPath)],shell=True)

	#Transfer the cropped images to a host
	def scp_cropped_images(self, hostName, imSz=256):
		print ('I AM HERE')
		fPaths  = sev2.get_folder_paths(self.id_, self.splitPrms_,
              self.isAlign_, hostName)
		tgHost  = scput.get_hostaddr(hostName)
		if self.isAlign_:
			srcPath  = self.paths_.crpImPathAlignTar % imSz
			tgPath   = tgHost + fPaths.crpImPathAlignTar % imSz
		else:
			srcPath  = self.paths_.crpImPathTar % imSz
			tgPath   = tgHost + fPaths.crpImPathTar % imSz
		print (tgPath) 
		subprocess.check_call(['rsync -ravz %s %s' % (srcPath, tgPath)],shell=True)

	def untar_cropped_images(self, imSz=256):
		if self.isAlign_:
			trFile  = self.paths_.crpImPathAlignTar % imSz
			drName  = self.paths_.crpImPathAlign % imSz
		else:
			trFile  = self.paths_.crpImPathTar % imSz
			drName  = self.paths_.crpImPath % imSz
		ou.mkdir(drName)
		subprocess.check_call(['tar -xf %s -C %s' % (trFile, drName)],shell=True)


def recompute_all(folderName):
	sf = StreetFolder(folderName)		
	print ('Recomputing prefix')
	sf._save_prefixes()
	print ('Recomputing Group Counts')
	sf._save_target_group_counts()
	print ('Recomputing Groups')
	sf._save_target_groups(forceWrite=True)	
	print ('Recomputing Splits')
	sf.split_trainval_sets()


def save_cropped_ims(args):
	folderName, isAligned = args
	sf = StreetFolder(folderName, isAlign=isAligned)	
	print ('Saving cropped images %s' % folderName)
	sf.save_cropped_images()

def tar_cropped_ims(args):
	folderName, isAligned = args
	sf = StreetFolder(folderName, isAlign=isAligned)	
	print ('Saving cropped images %s' % folderName)
	sf.tar_cropped_images()

def fetch_cropped_ims(args):
	folderName, isAligned, hostName, imSz = args
	sf = StreetFolder(folderName, isAlign=isAligned)	
	print ('Saving cropped images %s' % folderName)
	sf.fetch_scp_cropped_images(hostName, imSz=imSz)

def send_cropped_ims(args):
	folderName, isAligned, hostName, imSz = args
	sf = StreetFolder(folderName, isAlign=isAligned)	
	print ('Saving cropped images %s' % folderName)
	sf.scp_cropped_images(hostName, imSz=imSz)

#First form the groups
def save_groups(args):
	folderName, isAligned = args
	sf = StreetFolder(folderName, isAlign=isAligned)		
	print ('Saving groups for %s' % folderName)
	sf._save_target_groups()

#Second, save the splits
def save_trainval_splits(args):
	folderName, isAligned = args
	sf = StreetFolder(folderName, isAlign=isAligned)		
	print ('Saving splits for %s' % folderName)
	sf.split_trainval_sets()

#Thid, tar the trainval splits
def tar_trainval_splits(args):
	folderName, isAligned = args
	sf = StreetFolder(folderName, isAlign=isAligned)		
	print ('Saving splits for %s' % folderName)
	sf.tar_trainval_splits()

def send_trainval_splits(args):
	folderName, isAligned, hostName = args
	sf = StreetFolder(folderName, isAlign=isAligned)		
	print ('Sending splits for %s' % folderName)
	sf.scp_trainval_splits(hostName)

def fetch_trainval_splits(args):
	folderName, isAligned, hostName = args
	sf = StreetFolder(folderName, isAlign=isAligned)		
	print ('Sending splits for %s' % folderName)
	sf.fetch_scp_trainval_splits(hostName)

#Run functions in parallel that except a single argument folderName
def run_parallel(fnName, *args, **kwargs):
	dArgs = edict()
	dArgs.debugMode=False
	dArgs.isAligned=False
	kwargs = ou.get_defaults(kwargs, dArgs, True)
	args = list(args)
	if kwargs['debugMode']:
		#fNames = ['0070', '0071']
		fNames = ['0001', '0002']
		inArgs = [[osp.join('raw', f), kwargs['isAligned']] + args for f in fNames]
	else:
		listFile = 'geofence/dc-v2_list.txt'
		fid      = open(listFile, 'r')
		inArgs   = [[l.strip(), kwargs['isAligned']] + args for l in fid.readlines()]
		fid.close()
	print inArgs
	try:
		pool   = Pool(processes=6)
		jobs   = pool.map_async(fnName, inArgs)
		res    = jobs.get()
	except KeyboardInterrupt:
		raise Exception('Keyboard Interrupt')
		
	del pool


#### OLD NOT REQUIRED ####
def tar_folder_data(folderName):
	sf = StreetFolder(folderName)	
	drName  = sf.paths_.dr
	trFile  = sf.paths_.tarFile
	forceWrite = True
	if not osp.exists(trFile) or forceWrite:
		print ('Making %s' % trFile)
		subprocess.check_call(['tar -cf %s -C %s .' % (trFile, drName)],shell=True)
		return True
	else:
		print ('Already exists %s' % trFile)
		return False


