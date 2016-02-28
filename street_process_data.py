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

def get_config_paths():
	hostName = socket.gethostname()
	pths     = edict()
	if 'ivb' in hostName:
		pass
	else:
		pths.mainDataDr = '/data0/pulkitag/data_sets/streetview'
	pths.folderProc = osp.join(pths.mainDataDr, 'proc2', '%s')
	pths.cwd = osp.dirname(osp.abspath(__file__))
	return pths


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

	
def get_trainval_split_prms(**kwargs):
	dArgs = edict()
	dArgs.trnPct = 85
	dArgs.valPct = 5
	dArgs.tePct  = 10
	assert (dArgs.trnPct + dArgs.valPct + dArgs.tePct == 100)
	#The minimum distance between groups belonging to two sets
	#in meters (m)
	dArgs.minDist = 100
	dArgs = ou.get_defaults(kwargs, dArgs, True)
	dArgs.pStr = 'trn%d_val%d_te%d_dist%d' % (dArgs.trnPct, dArgs.valPct,
                dArgs.tePct, dArgs.minDist)
	return dArgs	


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
	def __init__(self, latitude, longitude, z=0):
		#In radians
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
		grp.num  = len(prefix)
		grp.prefix   = prefix
		grp.crpImNames    = crpImNames
		grp.data     = []
		grp.folderId = folderId
		grp.gid      = gId
		for i,p in enumerate(prefix):
			bsName = osp.basename(lbNames[i]).split('.')[0]
			bsGid  = bsName.split('_')[3]
			assert bsName == p, 'bsName:%s, p: %s'% (bsName,p)
			assert bsGid  == gId, 'Group name mismatch: %s, %s' % (bsGid, gid)
			grp.data.append(StreetLabel.from_file(lbNames[i]))
		return cls(grp)

	def distance_from_other(self, grp2):
		return get_distance_between_groups(self.grp, grp2.grp)
	
	def as_dict(self):
		grpDict = edict()
		for k in self.grp.keys():
			if k == 'data':
				continue
			grpDict[k] = self.grp[k]	
		grpDict['data'] = []
		for d in self.grp.data:
			grpDict['data'] = d.label
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
		print (cx, cy, self._gridCount_)
		self._gridCount_ += len(self.binList[cx][cy])
		breakFlag   = False
		if self._gridCount_ >= mxCount:
			breakFlag = True
		return breakFlag

	#Divide the group into two halves such that
	def divide_group_counts(self, mxCount):
		nX, nY = len(self.binList), len(self.binList[0])
		#Slowly prune the groups
		minY, maxY = 0, 1
		minX, maxX = 0, 1
		#Current y and x locations
		cy,  cx    = 0, 0
		self._gridCount_ = 0
		gBins      = []
		while True:
			for cy in range(minY, maxY):
				gBins.append((cx, cy))
				breakFlag = self._update_grid_count(cx, cy, mxCount)
				if breakFlag:
					break
			if breakFlag:
				break
			cy = min(nY - 1, maxY)
			for x	in range(0, maxX):
				gBins.append((cx, cy))
				breakFlag = self._update_grid_count(cx, cy, mxCount)
				cx += 1
				if breakFlag:
					break
			if breakFlag:
				break
			for y	in range(maxY, -1,-1):
				gBins.append((cx, cy))
				breakFlag = self._update_grid_count(cx, cy, mxCount)
				cy -= 1
				if breakFlag:
					break
			if breakFlag:
				break
			cy   = min(nY, maxY + 1)
			minY = cy
			cx = 0
			maxX = min(nX, maxX + 1)
			maxY = min(nY, maxY + 1)
			if maxX == nX or maxY == nY:
				print ('THIS CODE IS NOT TESTED when nX==maxX or ny == maxY,
					it works with a assumption that grid is square or rectangular
					CANNOT Trust code ..so raising exception')
				raise Exception('###### EXITING #####')
		#Find the bordering bins
		bBins = []
		#Find the bins along the x-axis
		for x in range(0, cx+2):
			bBins.append((x, min(nY-1, maxY+1)))
		if cx < maxX:
			for x in range(cx, maxX+1):
				bBins.append((x, min(nY-1, maxY)))
		for y in range(maxY, cy-1, -1):
			bBins.append((min(maxX + 1, nX-1), y))
		for y in range(cy, -1, -1):
			bBins.append((min(maxX, nX-1), y))
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
	def __init__(self, name, splitPrms=None):
		'''
			name:  relative path where the raw data in the folder is present
			svPth: the path where the processed data from the folder should be saved
		'''
		#folder id
		fStore        = FolderStore('streetview')
		self.id_      = fStore.get_id(name)
		self.name_    = name
		if splitPrms is None:
			self.splitPrms_ = get_trainval_split_prms() 
		#folder paths
		self._paths()
		#Prefixes
		self.prefixList_ = []
		self._read_prefixes_from_file()

	#
	def _paths(self):
		cPaths        = get_config_paths() 
		#raw data
		self.dirName_ = osp.join(cPaths.mainDataDr, self.name_)
		self.paths_   = edict()
		self.paths_.dr = cPaths.folderProc % self.id_
		ou.mkdir(self.paths_.dr)
		self.paths_.prefix     = osp.join(self.paths_.dr, 'prefix.pkl')
		self.paths_.prePerGrp  = osp.join(self.paths_.dr, 'prePerGrp.pkl')
		#List of targetgroups in ordered format - necessary as ordering matters
		#ordering can be used to split the data into train/val/test as points
		#closer in the ordering are physically close to each other
		self.paths_.targetGrpList = osp.join(self.paths_.dr, 'targetGrpList.pkl')
		self.paths_.targetGrps = osp.join(self.paths_.dr, 'targetGrps.pkl')
		#path for storing the cropped images
		self.paths_.crpImStr   = 'imCrop/imSz%s' % '%d'
		self.paths_.crpImPath  = osp.join(self.paths_.dr, self.paths_.crpImStr)
		#Split the sets
		self.paths_.trainvalSplit = osp.join(self.paths_.dr, 
                     'splits-%s.pkl' % self.splitPrms_.pStr)
		self.paths_.grpSplits  = edict()
		for s in ['train', 'val', 'test']:
			self.paths_.grpSplits[s]  = osp.join(self.paths_.dr, 
											 'groups_%s_%s.pkl' % (s, self.splitPrms_.pStr))


	#Save all prefixes in the folder
	def _save_prefixes(self):
		allNames = os.listdir(self.dirName_)
		imNames   = sorted([f for f in allNames if '.jpg' in f], reverse=True)
		lbNames   = sorted([f for f in allNames if '.txt' in f], reverse=True)
		prefixStr = []
		for (i,imn) in enumerate(imNames):
			imn = imn[0:-4]
			if i>= len(lbNames):
				continue
			if imn in lbNames[i]:
				prefixStr = prefixStr + [imn] 
		pickle.dump({'prefixStr': prefixStr}, open(self.paths_.prefix, 'w'))


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
		self._save_prefixes()	
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
		data = pickle.load(open(self.paths_.targetGrpList, 'r'))
		return data['grpList']	
		
	#save the target group data
	def _save_target_groups(self, forceWrite=False):
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

	#get groups
	def get_target_groups(self):
		dat = pickle.load(open(self.paths_.targetGrps, 'r'))
		return dat['groups']		

	#crop and save images - that makes it faster to read for training nets
	def save_cropped_images(self, imSz=256, isForceWrite=False):
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
		trnKeys, oKeys = gList.get_split_groups(nVal + nTe)
		print (N, len(trnKeys) + len(oKeys))
		print ('Num-Train: %d, Num-Others: %d, Total: %d' %
           (len(trnKeys), len(oKeys), len(gKeys)))
		return	

		firstValIdx  = nTrn + numSafety
		lastTrainGrp = grps[gKeys[nTrn]]
		print ('Determining validation groups')
		for n in range(firstValIdx, N):
			tDist = get_distance_between_groups(lastTrainGrp.grp, grps[gKeys[n]].grp)
			print (tDist)
			if tDist > sPrms.minDist:
				break
		firstValIdx = n+1
		trnKeys = [gKeys[i] for i in  range(0, nTrn)] 
		valKeys = [gKeys[i] for i in  range(firstValIdx, N)]
		print ('Num-Train: %d, Num-Val: %d' % 
           (len(trnKeys), len(valKeys)))
		print ('Pruning validation keys')
		setKeys['val']   = prune_groups(grps, trnKeys[::-1], valKeys, sPrms.minDist)
		return
	
		print ('Determining test groups')
		lastVal = n + nVal 
		last    = grps[gKeys[lastVal]]
		for n in range(lastVal + 1, N):
			tDist = get_distance_between_groups(last.grp, grps[gKeys[n]].grp)
			if tDist > sPrms.minDist:
				break
		teKeys  = [gKeys[i] for i in range(n, N)]
		print ('Num-Train: %d, Num-Val: %d, Num-Test: %d' % 
           (len(trnKeys), len(valKeys), len(teKeys)))
		#Greedily remove the common groups
		#train-val, train-test, val-test
		print ('Pruning validation keys')
		setKeys = edict()
		setKeys['train'] = trnKeys
		setKeys['val']   = prune_groups(grps, trnKeys[::-1], valKeys, sPrms.minDist)
		print ('Pruning test keys')
		setKeys['test']  = prune_groups(grps, setKeys['val'][::-1], teKeys, sPrms.minDist)
		print ('Pruning test keys, phase 2')
		setKeys['test'] = prune_groups(grps, trnKeys[::-1], setKeys['test'], sPrms.minDist)
		print ('Num-Train: %d, Num-Val: %d, Num-Test: %d' % 
           (len(setKeys['train']), len(setKeys['val']), len(setKeys['test'])))
		#Save the split keys
		pickle.dump({'setKeys': setKeys, 'splitPrms': self.splitPrms_},
               open(self.paths_.trainvalSplit, 'w')) 
		for s in ['train', 'val', 'test']:
			sGroups = [grps[gk].as_dict() for gk in setKeys[s]]
			pickle.dump({'groups': sGroups}, open(self.paths_.grpSplits[s], 'w')) 	
	

def save_processed_data(folderName):
	sf = StreetFolder(folderName)		
	#print ('Saving groups for %s' % folderName)
	#sf._save_target_groups()
	#print ('Saving splits for %s' % folderName)
	sf.split_trainval_sets()


def save_processed_data_multiple():
	fNames = ['0001', '0002', '0003', '0004', '0005']
	inArgs = [osp.join('raw', f) for f in fNames]
	for f in inArgs:
		sf = StreetFolder(f)		
	pool   = Pool(processes=6)
	jobs   = pool.map_async(save_processed_data, inArgs)
	res    = jobs.get()
	del pool	
