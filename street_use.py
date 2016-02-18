import numpy as np
import my_pycaffe_utils as mpu
import my_pycaffe_utils as mpio
from pycaffe_config import cfg
from os import path as osp
import street_test as st
import my_exp_pose as mepo
import street_exp as se
from easydict import EasyDict as edict
import os
import rot_utils as ru
import scipy.misc as scm
import copy
import math

class PoseCompute(object):
	def __init__(self, batchSz=1, modelIter=20000):
		prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1()
		exp         = se.setup_experiment(prms, cPrms)
		#Setup the Net
		mainDataDr = cfg.STREETVIEW_DATA_MAIN
		meanFile   = osp.join(mainDataDr,
								 'pulkitag/caffe_models/ilsvrc2012_mean_for_siamese.binaryproto')
		rootFolder = osp.join(mainDataDr,
								 'pulkitag/data_sets/streetview/proc/resize-im/im256/')
		batchSz    = batchSz
		testNet = mpu.CaffeTest.from_caffe_exp(exp)
		testNet.setup_network(opNames=['fc5'], imH=101, imW=101, cropH=101, cropW=101,
									channels = 6, chSwap=(2,1,0,5,4,3), 
									modelIterations=modelIter, delAbove='pose_fc', batchSz=batchSz,
									isAccuracyTest=False, dataLayerNames=['window_data'],
									newDataLayerNames = ['pair_data'],
									meanFile =meanFile)
		#Assing the net
		self.net_ = testNet

	def compute(self, im):
		'''
			im: should be HxWx6
		'''
		ims = im.reshape((1,) + im.shape)
		feats = self.net_.net_.forward_all(blobs=['pose_fc'], **{'pair_data': ims})
		predFeat = copy.deepcopy(feats['pose_fc'])
		euler = st.to_degrees(predFeat.squeeze())
		return euler


def get_kitti_paths():
	imDir = '/data1/pulkitag/data_sets/kitti/odometry'
	pth   = edict()
	pth['leftImFile']  = osp.join(imDir, 'dataset', 'sequences', '%02d','image_2','%06d.png')
	pth['rightImFile'] = osp.join(imDir, 'dataset', 'sequences', '%02d','image_3','%06d.png')
	pth['poseFile']    = os.path.join(imDir, 'dataset', 'poses', '%02d.txt')
	return pth

def get_kitti_images_seq(seqNum=None):
	#seq0: 
	#seq1: Driving on highway
	#seq2: Driving through countryside.  
	#seq3: Driving through countryside. 
	#seq4: City wide streets
	#seq5: Narrow streets within city and lots of houses
	#seq6: Similar to 5, but wider streets. 
	#seq7: Similar to 5 but a lot more other moving cars. 
  #seq8: Country Side and houses 
	#seq9: Country side and houses. More simialr to 8.
	#seq10: Narrow streets wihtin city + lots of trees and houses 
	#seq11: Narrow steets with houses + some narrow highway.
	allNum = [4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201]
	if seqNum is None:
		return allNum
	else:
		return allNum[seqNum]


def get_kitti_labels(pose, isRadian=False):
	rMat          = pose[:3,:3]
	lb1, lb2, lb3 = ru.mat2euler(rMat)
	if not isRadian:
		lbs = map(lambda x: x*180/np.pi, [lb1, lb2, lb3])
		return lbs
	else:
		return (lb1, lb2, lb3)


def get_kitti_rot_mats(seqNum=0):
	'''
		Provides the pose wrt to frame 1 in the form of (deltaX, deltaY, deltaZ, thetaZ, thetaY, thetaX
	'''
	if seqNum > 10 or seqNum < 0:
		raise Exception('Poses are only present for seqNum 0 to 10')

	paths  = get_kitti_paths()
	psFile = paths['poseFile'] % seqNum

	fid     = open(psFile, 'r')
	lines   = fid.readlines()
	allVals = np.zeros((len(lines), 3, 4)).astype(float)
	for (i,l) in enumerate(lines):
		vals      = [float(v) for v in l.split()]
		allVals[i]    = np.array(vals).reshape((3,4))
	fid.close()
	return allVals


def kitti_rotmat_diff(rDiff):
	tr = np.trace(rDiff)
	d  = 0.5*(tr - 1.0)
	return 180*math.acos(max(min(d, 1.0),-1.0))/np.pi
		

def estimate_odometry_kitti():
	pth     = get_kitti_paths() 
	seqNum  = 0
	pComp   = PoseCompute() 
	allMats = get_kitti_rot_mats(seqNum)
	r1      = np.eye(3) 
	for i in range(100):
		im1  = scm.imread(pth.leftImFile % (seqNum, i+1))
		im2  = scm.imread(pth.leftImFile % (seqNum, i))
		im   = np.concatenate((im1, im2), axis=2)
		pred = pComp.compute(im)
		mat  = ru.euler2mat(pred[0], pred[1], 0, isRadian=False)
		r1   = np.dot(mat, r1)
		thetas   = ru.mat2euler(r1)	
		print (map(lambda x: (x/np.pi) * 180, thetas))
		print (kitti_rotmat_diff(np.dot(r1.transpose(), allMats[i][:3,:3])))
		print(np.sqrt(reduce(lambda x, y: x*x + y*y, allMats[i][:,3])))
		print ('#######')	


