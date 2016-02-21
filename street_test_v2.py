##Latest version of results after correcting the bugs
#in the rotation computation

import street_exp as se
import my_pycaffe as mp
import my_pycaffe_utils as mpu
import my_pycaffe_io as mpio
import my_exp_v2 as mev2
import matplotlib.pyplot as plt
import vis_utils as vu
import numpy as np
import caffe
import copy
import os
from os import path as osp
import my_exp_pose_v2 as mepo2	
from transforms3d.transforms3d import euler as t3eu
from pycaffe_config import cfg
import scipy.misc as scm
import pickle
from scipy import io as sio
import street_test as ste

##
#Get the proto for pose regression	
def get_street_pose_proto(exp, protoType='all'):
	if protoType == 'mx90':
		wFile     = 'test-files/test_pose_euler_mx90_geo-dc-v2_spDist100_imSz256.txt'
		numIter   = 100
	elif protoType == 'all':
		wFile = 'test-files/test_pose-euler_spDist100_spVer-v1_geodc-v2_geo-dc-v2_lbNrmlz-zscore_crpSz192_nTe-1.00e+04_rawImSz256_exp-V2.txt'
		numIter   = 100
	netDef    = mpu.ProtoDef(exp.files_['netdef'])
	paramStr  = netDef.get_layer_property('window_data', 'param_str')[1:-1]
	paramStr  = ste.modify_params(paramStr, 'source', wFile)
	paramStr  = ste.modify_params(paramStr, 'batch_size', 100)
	netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % paramStr, phase='TEST')
	netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % paramStr)
	#If ptch loss is present
	lNames = netDef.get_all_layernames()
	if 'ptch_loss' in lNames:
		netDef.del_layer('ptch_loss')
		netDef.del_layer('ptch_fc')
		netDef.del_layer('slice_label')
		netDef.del_layer('accuracy')
		netDef.set_layer_property('window_data', 'top',
							'"%s"' % 'pose_label', phase='TEST', propNum=1)
		netDef.set_layer_property('window_data', 'top',
							'"%s"' % 'pose_label', propNum=1)
	defFile = 'test-files/pose_street_test.prototxt'
	netDef.write(defFile)
	return defFile, numIter

##
#Undo the normalization
def denormalize(prms, lbl, angleType='euler'):
	lbl       = copy.deepcopy(lbl)
	nrmlzFile = prms.paths.exp.window.nrmlz
	dat       = pickle.load(open(nrmlzFile, 'r'))
	if prms['lbNrmlz'] == 'zscore':
		mu, sd = dat['mu'][0:-1], dat['sd'][0:-1]
		print (mu, len(mu))
		assert lbl.shape[1] == len(mu), lbl.shape
		for lbIdx in range(len(mu)):
			lbl[:,lbIdx] = (lbl[:,lbIdx] * sd[lbIdx]) + mu[lbIdx]
	else:
		raise Exception ('Normalization not understood')	
	return lbl

##
#Determine the difference in rotations
def delta_rots(lbl1, lbl2, isOpRadian=False, opDeltaOnly=True):
	'''
		lbl1: assumed to be Nx3 or Nx2
					pitch, yaw, roll if 3
          pitch, yaw otherwise
		lbl2: same format as lbl1
		isOpRadian: True  - output in radians
								False - output in degrees
	'''
	N1, s1 = lbl1.shape
	assert s1 == 2 or s1 ==3
	N2, s2 = lbl2.shape
	assert N1==N2 and s1==s2
	if s1 == 2:
		p1, y1 = lbl1[:,0], lbl1[:,1]
		p2, y2 = lbl2[:,0], lbl2[:,1]
		r1, r2 = np.zeros((N1,)), np.zeros((N1,))
	else:
		p1, y1, r1 = lbl1[:,0], lbl1[:,1], lbl1[:,2]
		p2, y2, r2 = lbl2[:,0], lbl2[:,1], lbl2[:,2]
	deltaRot, rot1, rot2 = [], [], []
	for n in range(N1):
		rMat1      = t3eu.euler2mat(p1[n], y1[n], r1[n], 'szxy')
		rMat2      = t3eu.euler2mat(p2[n], y2[n], r2[n], 'szxy')
		dRotMat    = np.dot(rMat2, rMat1.transpose())
		pitch, yaw, roll = t3eu.mat2euler(dRotMat, 'szxy')
		_, dtheta  = t3eu.euler2axangle(pitch, yaw, roll, 'szxy')
		_, theta1  = t3eu.euler2axangle(p1[n], y1[n], r1[n], 'szxy') 
		_, theta2  = t3eu.euler2axangle(p2[n], y2[n], r2[n], 'szxy') 
		deltaRot.append(dtheta)
		rot1.append(theta1)
		rot2.append(theta2)
	if not isOpRadian:
		deltaRot = [(x * 180.)/np.pi for x in deltaRot]
		rot1     = [(x * 180.)/np.pi for x in rot1]
		rot2     = [(x * 180.)/np.pi for x in rot2]
	if opDeltaOnly:
		return deltaRot
	else:
		return deltaRot, rot1, rot2

	
##
#Test the pose net
def test_pose(prms, cPrms=None,  modelIter=None, protoType='all'):
	if cPrms is None:
		exp = prms
	else:
		exp       = se.setup_experiment(prms, cPrms)
	if protoType == 'pascal3d':
		defFile = exp.files_['netdef']
		numIter = 100
	else:
		defFile, numIter =  get_street_pose_proto(exp, protoType=protoType)
	modelFile = exp.get_snapshot_name(modelIter)
	caffe.set_mode_gpu()
	net = caffe.Net(defFile, modelFile, caffe.TEST)
	gtLabel, pdLabel, loss = [], [], []
	for i in range(numIter):
		data = net.forward(['pose_label','pose_fc', 'pose_loss'])
		gtLabel.append(copy.deepcopy(data['pose_label'][:,0:2].squeeze()))
		pdLabel.append(copy.deepcopy(data['pose_fc']))
		loss.append(copy.deepcopy(data['pose_loss']))
	gtLabel = denormalize(prms, np.concatenate(gtLabel))
	pdLabel = denormalize(prms, np.concatenate(pdLabel))
	lbInfo = prms.labels[0]
	if lbInfo.labelType_ in ['euler']:
		err, gtTheta, _ = delta_rots(gtLabel, pdLabel, opDeltaOnly=False) 	
	elif lbInfo.labelType_ in ['euler-5dof']:
		err, gtTheta, _ = delta_rots(gtLabel[:,0:2], pdLabel[:,0:2], opDeltaOnly=False) 	
	else:
		raise Exception ('LabelType %s not recognized' % lbInfo.labelType_)	
	medErr  = np.median(err, 0)
	muErr   = np.mean(err,0)
	return gtTheta, err

