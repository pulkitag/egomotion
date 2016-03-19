import pickle
import scipy.io as sio
import scipy.misc as scm
import numpy as np
import street_config as cfg
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import os
from os import path as osp
import other_utils as ou
import pascal_exp as pep
import subprocess
import pdb
from sklearn.cluster import KMeans

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

def get_paths():
	expDir, dataDir = cfg.pths.nyu.expDr, cfg.pths.nyu.dataDr
	pth = edict()	
	pth.exp = edict()
	pth.exp.dr= expDir
	#Snapshots
	pth.exp.snapshot    = edict()
	pth.exp.snapshot.dr = osp.join(pth.exp.dr, 'snapshot')
	ou.mkdir(pth.exp.snapshot.dr)
	#Get the label-stats
	pth.exp.labelStats  = osp.join(pth.exp.dr, 'label_stats.pkl')
	#Normal centers
	pth.exp.nrmlClusters = osp.join(pth.exp.dr, 'nrml_clusters.pkl')
	pth.exp.nrmlClustersReSz = osp.join(pth.exp.dr, 'nrml_clusters_resize.pkl')
	#info label for the experiment
	#pth.exp.lbInfo     = osp.join(pth.exp.dr, 'label_info', dPrms.expStr, 'lbinfo.pkl') 
	#ou.mkdir(osp.dirname(pth.exp.lbInfo))
	#Results
	pth.exp.results = edict()
	pth.exp.results.dr   = osp.join(pth.exp.dr, 'results', '%s')
	pth.exp.results.file = osp.join(pth.exp.results.dr, 'iter%d.pkl') 
	#Data files
	pth.data      = edict()
	pth.data.dr   = dataDir	
	pth.data.gtnrmlRaw = osp.join(dataDir, 'normals_gt', 'normals','%04d.png')
	pth.data.maskRaw   = osp.join(dataDir, 'normals_gt', 'masks','%04d.png')
	pth.data.gtnrml    = osp.join(dataDir, 'normals_gt_renamed', 'normals', '%04d.png')
	pth.data.imRaw     = osp.join(dataDir, 'ims', 'im%04d.jpg')
	pth.data.imFolder = osp.join(dataDir, 'imCrop', 'imSz%d_pad%d')
	#pth.data.imFolder = pth.data.imFolder % (dPrms.imCutSz, dPrms.imPadSz)
	#base net files
	pth.baseProto = osp.join(REAL_PATH, 'base_files', '%s.prototxt')
	#Window files
	windowDr      = osp.join(REAL_PATH, 'pose-files')
	pth.window  = edict()
	#Window files stores theta in degrees
	#pth.window.train = osp.join(windowDr, 'euler_train_pascal3d_imSz%d_pdSz%d.txt')
	#pth.window.test  = osp.join(windowDr, 'euler_test_pascal3d_imSz%d_pdSz%d.txt')
	#pth.window.train = pth.window.train % (dPrms.imCutSz, dPrms.imPadSz)
	#pth.window.test  = pth.window.test %  (dPrms.imCutSz, dPrms.imPadSz)
	return pth	

#original files are named starting with 0, rename them to start with 1
def rename_gt_normals(pths=None):
	if pths is None:
		pths = get_paths()
	for i in range(0,1449):
		rawName = pths.data.gtnrmlRaw % i
		finName = pths.data.gtnrml  % (i+1)		 
		subprocess.check_call(['cp %s %s' % (rawName, finName)], shell=True)

def show_images(pths=None):
	plt.ion()
	if pths is None:
		pths = get_paths()
	for i in range(100):
		im = scm.imread(pths.data.imRaw % (i+1))
		plt.imshow(im)		
		plt.show()
		plt.draw()
		ip = raw_input()
		if ip == 'q':
			return			

def blah():
	pass
	

def compute_normal_centers(isReSz=False):
	'''
		nrmls: N x 3 where N is the number of points
	'''
	pth       = get_paths()
	numIm     = 1449
	nSamples  = 20000
	K         = 20
	nrmls     = np.zeros((nSamples, 3))
	randState = np.random.RandomState(11)
	#Load all masks
	masks, ims = [], []
	for nn in range(numIm):
		mkName	= pth.data.maskRaw % nn
		mask    = scm.imread(mkName)
		mask    = mask[45:471, 41:601]
		if isReSz:
			mask  = cv2.resize(mask, (20,20))
		masks.append(mask.reshape((1,) + mask.shape))
		imName  = pth.data.gtnrmlRaw % nn
		im      = scm.imread(imName)/255.
		im      = im[45:471, 41:601,:]	
		if isReSz:
			im  = cv2.resize(im, (20,20))
		ims.append(im.reshape((1,) + im.shape))
	for ns in range(nSamples):
		while True:
			n = randState.randint(numIm)
			#Load he mask
			mask    = masks[n].squeeze()
			cmsm    = np.cumsum(mask)
			cmsm    = cmsm/float(cmsm[-1])
			rd      = randState.rand()
			try:
				idx     = pep.find_bin_index(cmsm, rd)
				yIdx, xIdx = np.unravel_index(idx, mask.shape)
			except:
				pdb.set_trace()
			#print (n, rd, idx)
			if not mask[yIdx][xIdx]:
				xIdx += 1
			if xIdx == mask.shape[1]:
				continue
			break
		assert mask[yIdx][xIdx], '%d, %d' % (yIdx, xIdx)
		#Load the image
		im      = ims[n].squeeze()
		nrl     = im[yIdx, xIdx,:].squeeze()
		sqSum   = np.sqrt(np.sum(nrl * nrl))
		nrl     = nrl / sqSum
		nrmls[ns,:] = nrl
	#K-Means clustering	
	cls = KMeans(n_clusters=20, random_state=randState)	
	cls.fit(nrmls)
	nrmlCenters = cls.cluster_centers_
	pickle.dump({'clusters': nrmlCenters}, open(pth.exp.nrmlClusters, 'w')) 


def load_clusters():
	pths = get_paths()
	dat  = pickle.load(open(pth.exp.nrmlClusters, 'r'))
	dat  = dat['clusters']
	Z    = np.sum(dat * dat, 1)
	dat  = dat / Z
	return dat
		

def get_cluster_index(dat, clusters):
	dist = clusters - dat
	dist = np.sum(dist * dist, 1)
	return np.argmin(dist)

		
def assign_normals_cluster(n, clusters=None):
	if clusters is None:
		clusters = load_clusters()
	pths = get_paths()
	nrmlFile = pth.data.gtnrmlRaw % n
	maskFile = pth.data.maskRaw % n
	nrml     = scm.imread(nrmlFile)
	mask     = scm.imread(maskFile)		
	mask     = mask[45:471, 41:601].astype(np.float32)
	nrml     = nrml[45:471, 41:601]/255.0
	mask     = cv2.resize(mask, [20 20])
	nrml     = cv2.resize(nrml, [20 20])
	mask     = mask > 0.5
	nrmlCluster = 20 * np.ones((20, 20))
	for i in range(20):
		for j in range(20): 
			if mask[i,j]:
				nrmlCluster[i,j] = get_cluster_index(nrml[i,j], clusters)
	return nrmlCluster			


def cluster2normals(nrmlCluster, clusters=None):
	if clusters is None:
		clusters = load_clusters()
	H, W  = nrmlCluster.shape	
	nrml = np.zeros((20,20,3)) 
	for i in range(H):
		for j in range(W):
			idx = nrmlClusters[i,j]
			if idx == 20:
				continue
			else:
				nrml[i,j,:] = clusters[idx,:]
	return nrml


def read_normal_file(fName):
	dat = sio.loadmat(sio)
	
