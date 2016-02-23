import read_liberty_patches as rlp
import my_exp_ptch as mept
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
import cv2
import my_exp_pose as mepo	
import street_cross_exp as sce
import rot_utils as ru
from pycaffe_config import cfg
import scipy.misc as scm
import pickle
from scipy import io as sio

def modify_params(paramStr, key, val):
	params = paramStr.strip().split('--')
	newStr = ''
	for i,p in enumerate(params):
		if len(p) ==0:
			continue
		if not(len(p.split()) == 2):
			continue
		k, v = p.split()
		if k.strip() == key:
			v = val
		newStr = newStr + '--%s %s ' % (k,v)
	return newStr

def get_fpr(recall, pdScore, gtLabel):
	pdScore = copy.deepcopy(pdScore)
	gtLabel = copy.deepcopy(gtLabel)
	N = sum(gtLabel==1)
	M = sum(gtLabel==0)
	assert(N+M == gtLabel.shape[0])
	idx = np.argsort(pdScore)
	#Sort in Decreasing Order
	pdScore = pdScore[idx[::-1]]
	gtLabel = gtLabel[idx[::-1]]
	posCount = np.cumsum(gtLabel==1)/float(N)
	threshIdx = np.where((posCount > recall)==True)[0][0]
	print (threshIdx, 'Thresh: %f' % pdScore[threshIdx])
	pdLabel   = pdScore >= pdScore[threshIdx]
	fp        = sum((pdLabel == 1) & (gtLabel == 0))
	tn        = sum((pdLabel == 0) & (gtLabel == 0))
	#pdLabel   = pdLabel[0:threshIdx]
	#gtLabel   = gtLabel[0:threshIdx]
	#err       = len(pdLabel) - np.sum(pdLabel==gtLabel)
	#fpr       = err/float(threshIdx)
	fpr        = float(fp)/float(fp + tn)
	return fpr
	
def get_liberty_ptch_proto(exp):
	libPrms   = rlp.get_prms()
	wFile     = libPrms.paths.wFile

	netDef    = mpu.ProtoDef(exp.files_['netdef'])
	paramStr  = netDef.get_layer_property('window_data', 'param_str')[1:-1]
	paramStr  = modify_params(paramStr, 'source', wFile)
	paramStr  = modify_params(paramStr, 'root_folder', libPrms.paths.jpgDir)
	paramStr  = modify_params(paramStr, 'batch_size', 100)
	netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % paramStr, phase='TEST')
	netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % paramStr)
	#If pose loss is present
	lNames = netDef.get_all_layernames()
	if 'pose_loss' in lNames:
		netDef.del_layer('pose_loss')
		netDef.del_layer('pose_fc')
		netDef.del_layer('slice_label')
		netDef.set_layer_property('window_data', 'top',
							'"%s"' % 'ptch_label', phase='TEST', propNum=1)
		netDef.set_layer_property('window_data', 'top',
							'"%s"' % 'ptch_label', propNum=1)
	defFile = 'test-files/ptch_liberty_test.prototxt'
	netDef.write(defFile)
	return defFile

def get_street_ptch_proto(exp, protoType='vegas'):
	if protoType == 'vegas':
		wFile     = 'test-files/vegas_ptch_test.txt'
		numIter   = 1000
	elif protoType == 'gt5':
		wFile     = 'test-files/ptch_test_euler-gt5.txt'
		numIter   = 90
	elif protoType == 'mxRot90':
		wFile   = 'test-files/test_ptch_mxRot90_equal-pos-neg_geo-dc-v2_spDist100_imSz256.txt'
		numIter  = 100
	elif protoType == 'newCity':
		wFile   = 'test-files/test_ptch_newcities.txt'
		numIter  = 100
	elif protoType == 'allRot':
		wFile    = 'test-files/test_ptch_equal-pos-neg_geo-dc-v2_spDist100_imSz256.txt'
		numIter  = 100
	else:
		raise Exception('%s not recognized' % protoType)
	netDef    = mpu.ProtoDef(exp.files_['netdef'])
	paramStr  = netDef.get_layer_property('window_data', 'param_str')[1:-1]
	paramStr  = modify_params(paramStr, 'source', wFile)
	paramStr  = modify_params(paramStr, 'batch_size', 100)
	netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % paramStr, phase='TEST')
	netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % paramStr)
	#If pose loss is present
	lNames = netDef.get_all_layernames()
	if 'pose_loss' in lNames:
		netDef.del_layer('pose_loss')
		netDef.del_layer('pose_fc')
		netDef.del_layer('slice_label')
		netDef.set_layer_property('window_data', 'top',
							'"%s"' % 'ptch_label', phase='TEST', propNum=1)
		netDef.set_layer_property('window_data', 'top',
							'"%s"' % 'ptch_label', propNum=1)
	defFile = 'test-files/ptch_street_test.prototxt'
	netDef.write(defFile)
	return defFile, numIter


def test_ptch(prms, cPrms=None, modelIter=None, protoType='vegas'):
	if cPrms is None:
		exp = prms
	else:
		exp       = se.setup_experiment(prms, cPrms)
	if protoType == 'liberty':
		defFile   = get_liberty_ptch_proto(exp)
		numIter   = 900
	else:
		defFile, numIter = get_street_ptch_proto(exp, protoType=protoType)
	modelFile = exp.get_snapshot_name(modelIter)
	caffe.set_mode_gpu()
	net = caffe.Net(defFile, modelFile, caffe.TEST)

	gtLabel, pdScore, acc = [], [], []
	for i in range(numIter):
		data = net.forward(['ptch_label','ptch_fc', 'accuracy'])
		print (sum(data['ptch_label'].squeeze()==1))
		gtLabel.append(copy.deepcopy(data['ptch_label'].squeeze()))
		score   = np.exp(data['ptch_fc'])
		score   = score/(np.sum(score,1).reshape(score.shape[0],1))
		pdScore.append(copy.deepcopy(score[:,1]))
		acc.append(copy.deepcopy(data['accuracy']))
	gtLabel = np.concatenate(gtLabel)
	pdScore = np.concatenate(pdScore)
	fpr     = get_fpr(0.95, copy.deepcopy(pdScore), copy.deepcopy(gtLabel))
	print 'FPR at 0.95 Recall is: %f' % fpr
	return gtLabel, pdScore

##
#Get the proto for pose regression	
def get_street_pose_proto(exp, protoType='mx90'):
	if protoType == 'mx90':
		wFile     = 'test-files/test_pose_euler_mx90_geo-dc-v2_spDist100_imSz256.txt'
		numIter   = 100
	elif protoType == 'all':
		wFile     = 'test-files/test_pose_euler_spDist100_geodc-v2_100K.txt'
		numIter   = 1000
	netDef    = mpu.ProtoDef(exp.files_['netdef'])
	paramStr  = netDef.get_layer_property('window_data', 'param_str')[1:-1]
	paramStr  = modify_params(paramStr, 'source', wFile)
	paramStr  = modify_params(paramStr, 'batch_size', 100)
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
#Convert the predictions to degrees
def to_degrees(lbl, angleType='euler', nrmlz=6.0/180.0):
	lbl = lbl/nrmlz
	return lbl

##
#
def get_rot_diff(gtLbl, pdLbl, angleType='euler', nrmlz=6.0/180.0):
	pdLbl = copy.deepcopy(pdLbl)
	gtLbl = copy.deepcopy(gtLbl)
	pdy = max(-180, pdLbl[0]/nrmlz)
	pdx = min(180, pdLbl[1]/nrmlz)
	mat1 = ru.euler2mat(0, gtLbl[0]/nrmlz, gtLbl[1]/nrmlz, isRadian=False)	
	mat2 = ru.euler2mat(0, pdy, pdx, isRadian=False)	
	rotDiff = np.dot(mat1,np.transpose(mat2))	
	theta, v = ru.rotmat_to_angle_axis(rotDiff)
	return (theta*180)/np.pi

##
def get_rot_diff_list(gtLbl, pdLbl, nrmlz=6.0/180.0):
	theta = []
	for (gt, pd) in zip(gtLbl, pdLbl):
		theta.append(get_rot_diff(gt, pd, nrmlz=nrmlz))
	return theta
	
##
#Get the distribution of errors
def get_binned_angle_errs(errs, angs):
	bins   = np.linspace(-180,180,20)
	mdErr  = np.zeros((len(bins)-1,))
	counts = np.zeros((len(bins)-1,))
	for i,bn in enumerate(bins[0:-1]):
		stVal = bn
		enVal = bins[i+1]
		print (stVal, enVal)
		idx   = (angs >= stVal) & (angs < enVal)
		counts[i] = np.sum(idx)
		#idx   = angs < enVal
		if np.sum(idx)>0:
			mdErr[i] = np.median(errs[idx]) 	
	return mdErr, counts

##
#Test the pose net
def test_pose(prms, cPrms=None,  modelIter=None, protoType='mx90'):
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
	gtLabel = to_degrees(np.concatenate(gtLabel))
	pdLabel = to_degrees(np.concatenate(pdLabel))
	err     = np.abs(gtLabel - pdLabel)
	medErr  = np.median(err, 0)
	muErr   = np.mean(err,0)
	return gtLabel, pdLabel, err

def vis_liberty_ptch():
	libPrms   = rlp.get_prms()
	wFile     = libPrms.paths.wFile
	wDat      = mpio.GenericWindowReader(wFile)
	rootDir   = libPrms.paths.jpgDir
	plt.ion()
	fig = plt.figure()
	while True:
		imNames, lbs = wDat.read_next()
		imNames  = [osp.join(rootDir, n.split()[0]) for n in imNames]
		figTitle = '%d' % lbs[0][0]
		im1      = plt.imread(imNames[0])
		im2      = plt.imread(imNames[1])
		vu.plot_pairs(im1, im2, fig=fig, figTitle=figTitle)	
		inp = raw_input('Press a key to continue')
		if inp=='q':
			return
	
def make_pascal3d_generic_window():
	srcFile  = '/data1/pulkitag/data_sets/pascal_3d/my/window_file_%s.txt'
	outFile  = '/data1/pulkitag/data_sets/pascal_3d/my/generic_window_file_%s.txt'
	setNames = ['train', 'val']
	for s in setNames:
		iFile = srcFile % s
		oFile = outFile % s
		with open(iFile) as fi:
			lines = fi.readlines()
			N     = len(lines)
			oFid  = mpio.GenericWindowWriter(oFile, N, 1, 3)
			for i,l in enumerate(lines[0:100]):
				if np.mod(i,1000)==1:
					print (i)
				print l
				fName, y1, y2, x1, x2, az, el, cl = l.strip().split()
				im      = cv2.imread(fName)
				h, w, ch = im.shape
				fSplit = fName.split('/')
				assert fSplit[-3] == 'Images'
				fName = fSplit[-2] + '/' + fSplit[-1]
				y1, y2, x1, x2 = float(y1), float(y2), float(x1), float(x2)
				az, el = float(az), float(el)
				cl     = int(cl)
				imLine = [[fName, [ch, h, w], [x1, y1, x2, y2]]]
				oFid.write([az, el, cl], *imLine)


def get_ptch_test_results_conv4():
	numConv4  = [8, 16, 32]
	modelIter = 26000
	fpr       = []
	for n in numConv4:
	 	prms, cPrms = mept.smallnetv6_pool4_ptch_crp192_rawImSz256(numConv4=n)
		gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, isLiberty=False)
		fpr.append(get_fpr(0.95, pdScore, gtLabel))
	return fpr
 
def get_ptch_test_results_fc5(protoType='gt5'):
	#numFc5    = [32, 64, 128, 256, 384, 512, 1024]
	#runNum    = [0, 0, 0, 1, 0, 0, 0]
	numFc5    = [128, 256, 384, 512, 1024]
	runNum    = [0, 1, 0, 0, 0]
	#numFc5    = [512]
	#runNum    = [0]
	modelIter = 72000
	fpr       = {}
	for n,r in zip(numFc5, runNum):
		try:
			if n in [128, 256, 384]:
				prms, cPrms = mept.smallnetv2_fc5_ptch_crp192_rawImSz256(numFc5=n, runNum=r)
			else:
				prms, cPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(numFc5=n, runNum=r)
			gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, 
																		protoType=protoType)
			fpr['num-%d' % n] = get_fpr(0.95, pdScore, gtLabel)
		except:
			print ('Not found for %d' % n)
	return fpr

##
#Get the results with the constrain that maximum allowed rotation is 90
def get_ptch_test_results_fc5_mxRot90():
	#numFc5    = [128, 256, 384, 512, 1024]
	numFc5    = [384, 512, 1024]
	runNum    = [0, 0, 0, 0, 0]
	modelIter = 72000
	fpr       = {}
	for n,r in zip(numFc5, runNum):
		try:
			print ('LO')
			prms, cPrms = mept.ptch_from_ptch_pose_euler_mx90_smallnetv5_fc5_exp1(numFc5=n)
			print ('Here')
			exp = se.setup_experiment(prms, cPrms)
			#print(osp.exists(exp.get_snapshotname(modelIter)))
			gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, isLiberty=False)
			fpr['num-%d' % n] = get_fpr(0.95, pdScore, gtLabel)
		except:
			print ('Not found for %d' % n)
	return fpr

def get_multiloss_on_ptch_results(protoType='mx90'):
	fpr = {}
	modelIter = 72000
	#With Conv4
	#prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v6_pool4_exp1(numConv4=32)
	#gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, isLiberty=False)
	#fpr.append(get_fpr(0.95, pdScore, gtLabel))

	#With Fc5 
	numFc = [128, 256, 384, 512, 1024]
	#numFc = [384, 1024]
	numFc = [512]
	for n in numFc:
		#prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v5_fc5_exp1(numFc5=n)
		prms, cPrms = mev2.ptch_pose_euler_smallnet_v5_fc5_exp1_lossl1(numFc5=None)
		#prms, cPrms = mev2.ptch_pose_euler_(numFc5=None)
		#try:
		gtLabel, pdScore = test_ptch(prms, cPrms, modelIter,
												protoType=protoType)
		fpr['num-%d' % n] = get_fpr(0.95, pdScore, gtLabel)
		#except:
		print ('Not found for %d' % n)
	return fpr

def get_pose_on_pose_results():
	medErr = {}
	modelIter = 72000

	#With Fc5 
	numFc = [128, 384, 512, 1024]
	#numFc = [32, 64]
	for n in numFc:
		try:
			#prms, cPrms = mepo.smallnetv5_fc5_pose_euler_mx90_crp192_rawImSz256(numFc5=n)
			prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256(numFc5=n)
			_, _, err   = test_pose(prms, cPrms, modelIter)
			medErr['num-%d' % n] = np.median(err,0)
		except:
			print ('NOT FOUND: %d' % n)
	return medErr

def test_linear_pose_from_ptch():
	exp = sce.train_pose_using_ptch()
	modelIter=40000		
	_, _, err   = test_pose(exp, None, modelIter)
	return np.median(err,0)

def test_linear_ptch_from_pose(protoType='gt5'):
	exp = sce.train_ptch_using_pose()
	modelIter=26000		
	gt, pd   = test_ptch(exp, None, modelIter, protoType=protoType)
	return get_fpr(0.95, pd, gt)

def test_linear_ptch_from_pose_all(protoType='gt5'):
	exp       = sce.train_ptch_using_pose_fc5()
	modelIter = 60000		
	gt, pd    = test_ptch(exp, None, modelIter, protoType=protoType)
	return get_fpr(0.95, pd, gt)


def get_multiloss_on_pose_results():
	medErr = {}
	modelIter = 40000
	#With Conv4
	#prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v6_pool4_exp1(numConv4=32)
	#gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, isLiberty=False)
	#fpr.append(get_fpr(0.95, pdScore, gtLabel))

	#With Fc5 
	numFc = [128, 256, 384, 1024]
	#numFc = [32, 64]
	for n in numFc:
		prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v5_fc5_exp1(numFc5=n)
		_, _, err   = test_pose(prms, cPrms, modelIter)
		medErr['num-%d' % n] = np.median(err,0)
	return medErr

def test_ptch_lt_euler_5(protoType='gt5'):
	numFc = [128, 512]
	fpr = {}
	modelIter = 36000
	for n in numFc:
		#prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v5_fc5_exp1(numFc5=n)
		prms, cPrms = mept.smallnetv5_fc5_ptch_euler_mx5_crp192_rawImSz256(numFc5=n)
		#exp = se.setup_experiment(prms,cPrms)
		#modelFile = exp.get_snapshot_name(modelIter)
		#print (osp.exists(modelFile))
		#continue
		try:
			gtLabel, pdScore = test_ptch(prms, cPrms, modelIter,
									 isLiberty=False, protoType=protoType)
			fpr['num-%d' % n] = get_fpr(0.95, pdScore, gtLabel)
		except:
			print ('Not found for %d' % n)
	return fpr

def test_ptch_lt_euler_5_pose_all(protoType='gt5'):
	numFc = [128, 512]
	fpr = {}
	modelIter = 36000
	for n in numFc:
		#prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v5_fc5_exp1(numFc5=n)
		prms, cPrms = mev2.ptch_euler_mx5_pose_euler_smallnet_v5_fc5_exp1(numFc5=n)
		#exp = se.setup_experiment(prms,cPrms)
		#modelFile = exp.get_snapshot_name(modelIter)
		#print (osp.exists(modelFile))
		#continue
		try:
			gtLabel, pdScore = test_ptch(prms, cPrms, modelIter,
											 isLiberty=False, protoType=protoType)
			fpr['num-%d' % n] = get_fpr(0.95, pdScore, gtLabel)
		except:
			print ('Not found for %d' % n)
	return fpr

def test_linear_ptch_from_ptch_lt5(protoType='gt5'):
	exp = sce.train_ptch_using_ptch_lt5()
	modelIter=46000		
	fpr  = test_ptch(exp, None, modelIter, protoType=protoType)
	return fpr

def test_linear_ptch_from_ptch_lt5_pose_all(protoType='gt5'):
	exp = sce.train_ptch_using_ptch_lt5_pose_all()
	modelIter=46000		
	fpr  = test_ptch(exp, None, modelIter, protoType=protoType)
	return fpr

##
#Save the York features
def save_york_feats(modelIter=20000, dataset='york', lossType='pose_l1', isMat=False):
	if lossType == 'pose_l2':
		prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256(numFc5=512)
		delLayers   = ['slice_pair']
	elif lossType == 'pose_l1':
		prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1()
		delLayers   = ['slice_pair']
	elif lossType == 'pose_ptch_l1':
		prms, cPrms = mev2.ptch_pose_euler_smallnet_v5_fc5_exp1_lossl1()
		delLayers   = ['slice_pair', 'slice_label']
	exp         = se.setup_experiment(prms, cPrms)
	#Read the images
	dataDir = '/work5/pulkitag/data_sets/streetview'
	if dataset == 'york':
		dirName = '/work5/pulkitag/data_sets/streetview/york/York_VP_imOnly'
		outDir = '/work5/pulkitag/data_sets/streetview/york/feats/'
	elif dataset == 'york_others':
		dirName = '/work5/pulkitag/data_sets/streetview/york_and_others/tvp_pku_york_ims'
		outDir = '/work5/pulkitag/data_sets/streetview/york_and_others/feats/'
	elif dataset == 'gen_test':
		dirName = '/work5/pulkitag/data_sets/streetview/gen_test/gen_test_imgs'
		outDir = '/work5/pulkitag/data_sets/streetview/gen_test/feats/'
	elif dataset == 'army':
		dirName = osp.join(dataDir, 'army_patches', 'Army_Patches')
		outDir  = osp.join(dataDir, 'army_patches', 'feats')
	elif dataset == 'places':
		dirName = osp.join(dataDir, 'places', 'images')
		outDir  = osp.join(dataDir, 'places', 'feats-%s' % lossType)
	elif dataset == 'lib_indoor':
		dirName = osp.join(dataDir, 'places', 'library_indoor', 'indoor')
		outDir  = osp.join(dataDir, 'places', 'library_indoor', 'feats-%s' % lossType)
	elif dataset == 'swim':
		dirName = osp.join(dataDir, 'places', 'swim_hotel', 'swimming_pool_indoor')
		outDir  = osp.join(dataDir, 'places', 'swim_hotel', 'swim-feats-%s' % lossType)
	elif dataset == 'room':
		dirName = osp.join(dataDir, 'places', 'swim_hotel', 'hotel_room')
		outDir  = osp.join(dataDir, 'places', 'swim_hotel', 'hotel-feats-%s' % lossType)
	elif dataset == 'herz':
		dirName = osp.join(dataDir, 'herz', 'jesuoutLarge2')
		outDir  = osp.join(dataDir, 'herz', 'herz-feats-%s' % lossType)
	elif dataset == 'pascal3d':
		dirName = osp.join(dataDir, 'pascal3d', 'val_crops')
		outDir  = osp.join(dataDir, 'pascal3d', 'val_feats-%s' % lossType)
	if not osp.exists(outDir):
		os.makedirs(outDir)	
	prefix = [f[0:-4] for f in os.listdir(dirName) if '.jpg' in f]
	imFile  = [osp.join(dirName, p + '.jpg') for p in prefix]
	if isMat:
		outName = [osp.join(outDir,  p + '.mat') for p in prefix]  
	else:
		outName = [osp.join(outDir,  p + '.pkl') for p in prefix]  

	#Setup the Net
	mainDataDr = cfg.STREETVIEW_DATA_MAIN
	meanFile   = osp.join(mainDataDr, 'pulkitag/caffe_models/ilsvrc2012_mean.binaryproto')
	batchSz    = 64
	testNet = mpu.CaffeTest.from_caffe_exp(exp)
	testNet.setup_network(opNames=['fc5'], imH=101, imW=101, cropH=101, cropW=101,
								modelIterations=modelIter, delAbove='relu5', batchSz=batchSz,
								isAccuracyTest=False, dataLayerNames=['window_data'], 
								newDataLayerNames=['data'], delLayers=delLayers,
								meanFile =meanFile)
	#Send images and get features
	for st in range(0,len(imFile),batchSz):
		en = min(len(imFile), st+ batchSz)
		ims      = []
		skipList = []
		for i in range(st,en):
			im = scm.imread(imFile[i])
			im = scm.imresize(im, (192,192))
			if im.ndim ==2:
				skipList.append(i)
				print ('SKIPPING GRAY SCALE IMAGE')
				continue
			if im.ndim==3 and im.shape[2] == 1:
				skipList.append(i)
				print ('SKIPPING GRAY SCALE IMAGE')
				continue
			ims.append(im.reshape((1,) + im.shape))
		ims = np.concatenate(ims)
		print ims.shape
		feats = testNet.net_.forward_all(blobs=['fc5'], **{'data': ims})
		count = 0
		for i in range(st,en):
			if i in skipList:
				continue
			if isMat:
				sio.savemat(outName[i], {'feat': feats['fc5'][count]})
			else:
				pickle.dump({'feat': feats['fc5'][count]}, open(outName[i], 'w'))
			count += 1	


def get_nrml_results(modelIter=20000):
	#prms, cPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256()
	prms, cPrms = mept.smallnetv2_pool4_ptch_crp192_rawImSz256(isPythonLayer=True)
	exp         = se.setup_experiment(prms, cPrms)
	#Setup the Net
	mainDataDr = cfg.STREETVIEW_DATA_MAIN
	meanFile   = osp.join(mainDataDr, 'pulkitag/caffe_models/ilsvrc2012_mean_for_siamese.binaryproto')
	batchSz    = 500
	testNet = mpu.CaffeTest.from_caffe_exp(exp)
	testNet.setup_network(opNames=['ptch_fc'], imH=101, imW=101, cropH=101, cropW=101,
								modelIterations=modelIter, delAbove='ptch_fc', batchSz=batchSz,
								channels = 6, chSwap=(2,1,0,5,4,3), 
								isAccuracyTest=False, dataLayerNames=['window_data'], 
								newDataLayerNames=['pair_data'],
								meanFile =meanFile)

	#Just read the first 10K images
	wFile = prms.paths['windowFile']['train']
	wFid  = mpio.GenericWindowReader(wFile)
	rootFolder='/data0/pulkitag/data_sets/streetview/proc/resize-im/im256'
	ims, lbs = [], []
	#N, numTest = 10000, 1000
	N, numTest = 100, 10
	for i in range(N):
		im, lb = wFid.read_next_processed(rootFolder)
		ims.append(im[0])
		lbs.append(lb)	

	#Read the test data
	wFile = prms.paths['windowFile']['test']
	wFid  = mpio.GenericWindowReader(wFile)
	rootFolder='/data0/pulkitag/data_sets/streetview/proc/resize-im/im256'
	gtIms, gtLbs = [], []
	for i in range(numTest):
		im, lb = wFid.read_next_processed(rootFolder)
		gtIms.append(im[0])
		gtLbs.append(lb)	

	#return ims, gtIms
	predFeat = []
	for te in range(numTest):
		imTe = gtIms[te].reshape((1,) + gtIms[te].shape)
		for tr in range(0,N,batchSz):
			batchIm = []
			for i in range(tr, min(tr+batchSz,N)):
				im = ims[i].reshape((1,) + ims[i].shape)
				batchIm.append(np.concatenate([imTe, im], axis=3))
			batchIm = np.concatenate(batchIm, axis=0)
			feats = testNet.net_.forward_all(blobs=['ptch_fc'], **{'pair_data': batchIm})
			predFeat.append(copy.deepcopy(feats['ptch_fc']))
	return predFeat

##
#Verify pose fc
def verify_pose_results(modelIter):
	#prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256(numFc5=512)
	prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1()
	exp         = se.setup_experiment(prms, cPrms)
	#Window File
	wFileName   = 'test-files/test_pose_euler_mx90_geo-dc-v2_spDist100_imSz256.txt'
	wFile       = mpio.GenericWindowReader(wFileName)
	#Setup the Net
	mainDataDr = cfg.STREETVIEW_DATA_MAIN
	meanFile   = osp.join(mainDataDr,
							 'pulkitag/caffe_models/ilsvrc2012_mean_for_siamese.binaryproto')
	rootFolder = osp.join(mainDataDr,
							 'pulkitag/data_sets/streetview/proc/resize-im/im256/')
	batchSz    = 100
	testNet = mpu.CaffeTest.from_caffe_exp(exp)
	testNet.setup_network(opNames=['fc5'], imH=101, imW=101, cropH=101, cropW=101,
								channels = 6, chSwap=(2,1,0,5,4,3), 
								modelIterations=modelIter, delAbove='pose_fc', batchSz=batchSz,
								isAccuracyTest=False, dataLayerNames=['window_data'],
								newDataLayerNames = ['pair_data'],
								meanFile =meanFile)
	predFeat, gtFeat = [], []
	#Send images and get features
	for st in range(0,200,batchSz):
		en = min(200, st+ batchSz)
		ims = []
		for i in range(st,en):
			im, lbls = wFile.read_next_processed(rootFolder)	
			im = np.concatenate(im, axis=2)
			ims.append(im.reshape((1,) + im.shape))
			gtFeat.append(lbls[0:2].reshape((1,) + lbls[0:2].shape))
		ims = np.concatenate(ims)
		print ims.shape
		feats = testNet.net_.forward_all(blobs=['pose_fc'], **{'pair_data': ims})
		predFeat.append(copy.deepcopy(feats['pose_fc']))
	gtFeat = np.concatenate(gtFeat)
	predFeat = np.concatenate(predFeat)
	err = np.median((np.abs(gtFeat - predFeat) * 30),axis=0)
	print (err)
	return gtFeat, predFeat

def save_alexnet_york_feats(dataset = 'york'):
	#Read the images
	dataDir = '/work5/pulkitag/data_sets/streetview'
	if dataset == 'york':
		dirName = '/work5/pulkitag/data_sets/streetview/york/York_VP_imOnly'
		outDir = '/work5/pulkitag/data_sets/streetview/york/feats-alexnet/'
	elif dataset == 'york_others':
		dirName = '/work5/pulkitag/data_sets/streetview/york_and_others/tvp_pku_york_ims'
		outDir = '/work5/pulkitag/data_sets/streetview/york_and_others/feats-alexnet/'
	elif dataset == 'gen_test':
		dirName = '/work5/pulkitag/data_sets/streetview/gen_test/gen_test_imgs'
		outDir = '/work5/pulkitag/data_sets/streetview/gen_test/feats-alexnet/'
	elif dataset == 'places':
		dirName = osp.join(dataDir, 'places', 'images')
		outDir  = osp.join(dataDir, 'places', 'feats-alexnet')
	if not osp.exists(outDir):
		os.makedirs(outDir)
	prefix = [f[0:-4] for f in os.listdir(dirName) if '.jpg' in f]
	imFile  = [osp.join(dirName, p + '.jpg') for p in prefix]
	outName = [osp.join(outDir,  p + '.pkl') for p in prefix]  

	#Setup the Net
	mainDataDr = cfg.STREETVIEW_DATA_MAIN
	meanFile   = osp.join(mainDataDr, 'pulkitag/caffe_models/ilsvrc2012_mean.binaryproto')
	netFile    = osp.join(mainDataDr, 
								'pulkitag/caffe_models/bvlc_reference/bvlc_reference_caffenet.caffemodel')
	defFile    = osp.join(mainDataDr, 
								'pulkitag/caffe_models/bvlc_reference/caffenet_full_deploy.prototxt')
	batchSz    = 256
	testNet    = mp.MyNet(defFile, netFile, caffe.TEST)
	testNet.set_preprocess(ipName = 'data', isBlobFormat=False,
				imageDims = (227, 227, 3),
				cropDims  = (227, 227), chSwap=(2,1,0),
				rawScale = None, meanDat = meanFile)

	#Send images and get features
	for st in range(0,len(imFile),batchSz):
		en = min(len(imFile), st+ batchSz)
		ims = []
		skipList = []
		for i in range(st,en):
			im = scm.imread(imFile[i])
			im = scm.imresize(im, (227,227))
			if im.ndim ==2:
				skipList.append(i)
				print ('SKIPPING GRAY SCALE IMAGE')
				continue
			if im.ndim==3 and im.shape[2] == 1:
				skipList.append(i)
				print ('SKIPPING GRAY SCALE IMAGE')
				continue
			ims.append(im.reshape((1,) + im.shape))
		ims = np.concatenate(ims)
		print ims.shape
		feats = testNet.forward_all(blobs=['fc7'], **{'data': ims})
		count = 0
		for i in range(st,en):	
			#pickle.dump({'pool5': feats['pool5'][i-st], 'fc6': feats['fc6'][i-st], 
			#						 'fc7': feats['fc7'][i-st]}, open(outName[i], 'w'))
			if i in skipList:
				continue	
			pickle.dump({'fc7': feats['fc7'][count]}, open(outName[i], 'w'))
			count += 1	
