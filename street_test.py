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
from os import path as osp
import cv2
import my_exp_pose as mepo	
import street_cross_exp as sce

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
	pdLabel   = pdLabel[0:threshIdx]
	gtLabel   = gtLabel[0:threshIdx]
	err       = len(pdLabel) - np.sum(pdLabel==gtLabel)
	fpr       = err/float(threshIdx)
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
	defFile = 'test-files/ptch_liberty_test.prototxt'
	netDef.write(defFile)
	return defFile

def get_street_ptch_proto(exp):
	#wFile     = 'test-files/test_ptch_equal-pos-neg_geo-dc-v2_spDist100_imSz256.txt'
	wFile     = 'test-files/test_ptch_mxRot90_equal-pos-neg_geo-dc-v2_spDist100_imSz256.txt'
	#wFile     = 'test-files/test_ptch_newcities.txt'
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
	return defFile


def test_ptch(prms, cPrms=None, modelIter=None, isLiberty=False):
	if cPrms is None:
		exp = prms
	else:
		exp       = se.setup_experiment(prms, cPrms)
	if isLiberty:
		defFile   = get_liberty_ptch_proto(exp)
		numIter   = 900
	else:
		defFile   = get_street_ptch_proto(exp)
		numIter   = 100
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
	if 'pose_loss' in lNames:
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
	if angleType =='euler':
		lbl = lbl/nrmlz
	return lbl

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
	defFile, numIter =  get_street_pose_proto(exp, protoType=protoType)
	modelFile = exp.get_snapshot_name(modelIter)
	caffe.set_mode_gpu()
	net = caffe.Net(defFile, modelFile, caffe.TEST)
	gtLabel, pdLabel = [], []
	for i in range(numIter):
		data = net.forward(['pose_label','pose_fc'])
		gtLabel.append(copy.deepcopy(data['pose_label'][:,0:2].squeeze()))
		pdLabel.append(copy.deepcopy(data['pose_fc']))
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
 
def get_ptch_test_results_fc5():
	numFc5    = [32, 64, 128, 256, 384, 512, 1024]
	runNum    = [0, 0, 0, 1, 0, 0, 0]
	#numFc5    = [4, 16]
	#runNum    = [0, 0, 0, 1, 0, 0, 0]
	modelIter = 72000
	fpr       = {}
	for n,r in zip(numFc5, runNum):
		try:
			if n in [128, 256, 384]:
				prms, cPrms = mept.smallnetv2_fc5_ptch_crp192_rawImSz256(numFc5=n, runNum=r)
			else:
				prms, cPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(numFc5=n, runNum=r)
			gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, isLiberty=False)
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



def get_multiloss_on_ptch_results():
	fpr = {}
	modelIter = 100000
	#With Conv4
	#prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v6_pool4_exp1(numConv4=32)
	#gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, isLiberty=False)
	#fpr.append(get_fpr(0.95, pdScore, gtLabel))

	#With Fc5 
	#numFc = [128, 256, 384, 1024]
	numFc = [384, 1024]
	#numFc = [512]
	for n in numFc:
		prms, cPrms = mev2.ptch_pose_euler_mx90_smallnet_v5_fc5_exp1(numFc5=n)
		try:
			gtLabel, pdScore = test_ptch(prms, cPrms, modelIter, isLiberty=False)
			fpr['num-%d' % n] = get_fpr(0.95, pdScore, gtLabel)
		except:
			print ('Not found for %d' % n)
	return fpr

def get_pose_on_pose_results():
	medErr = {}
	modelIter = 40000

	#With Fc5 
	numFc = [128, 384, 512, 1024]
	#numFc = [32, 64]
	for n in numFc:
		try:
			prms, cPrms = mepo.smallnetv5_fc5_pose_euler_mx90_crp192_rawImSz256(numFc5=n)
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

def test_linear_ptch_from_pose():
	exp = sce.train_ptch_using_pose()
	modelIter=72000		
	gt, pd   = test_ptch(exp, None, modelIter)
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
