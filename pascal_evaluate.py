import scipy.misc as scm
import pascal_exp_run as per
from pascal3d_eval import poseBenchmark as pbench
import pascal_exp as pep
import setup_pascal3d as sp3d
import cv2
import numpy as np
from os import path as osp
import my_pycaffe as mp
import copy
import pdb
import matplotlib.pyplot as plt
import pickle
from os import path as osp
from easydict import EasyDict as edict
import other_utils as ou
import math
from collections import OrderedDict
import pdb

PASCAL_CLS = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
              'chair', 'diningtable', 'motorbike', 'sofa', 'train',
              'tvmonitor']

def get_result_filename(exp, numIter):
	resFile = exp.dPrms_.paths.exp.results.file
	resFile = osp.join(resFile % (osp.join(exp.cPrms_.expStr, exp.dPrms_.expStr), numIter))
	ou.mkdir(osp.dirname(resFile))
	return resFile

#Get the objec that would be used to generate the benchmark data
def get_car_bench_obj():
	bench = pbench.PoseBenchmark('car')
	return bench

def get_exp(expNum=0, numIter=None):
	#2 Dropout, lower learning rate
	if expNum == 0:
		exp  = per.doublefcv1_dcv2_dof2net_cls_pd36(nElBins=21, nAzBins=21,
          crpSz=224, isDropOut=True, numDrop=2, base_lr=0.0001)
		expIter = 14000
	#2 Dropouts, 0.001 learning rate
	elif expNum == 1:
		exp = per.doublefcv1_dcv2_dof2net_cls_pd36(nElBins=21, nAzBins=21,
           crpSz=240, isDropOut=True, numDrop=2)
		expIter = 26000
	#1 Dropout, 0.001 learning rate
	elif expNum == 2:
		exp = per.doublefcv1_dcv2_dof2net_cls_pd36(nElBins=21, nAzBins=21,
           crpSz=224, isDropOut=True)	
		expIter = 12000
	#Tune only the last layer
	elif expNum == 3:
		exp = per.doublefcv1_dcv2_dof2net_cls_pd36(lrAbove='fc6')
		expIter = 40000
	#Torch-net
	elif expNum == 4:
		exp = exp = per.torchnet_cls_pd36()
		expIter = 40000
	#AlexNet
	elif expNum == 5:
		exp = per.alexnet_cls_pd36(nElBins=21, nAzBins=21, crpSz=224)
		expIter = 30000
	#Scratch, double dropout
	elif expNum == 6:
		exp = per.scratch_cls_pd36(nElBins=21, nAzBins=21, 
        isDropOut=True, numDrop=2)
		expIter = 12000
	#slower learning rate for finetuning
	elif expNum == 7:
		exp = per.doublefcv1_dcv2_dof2net_cls_pd36(nElBins=21, nAzBins=21,
          crpSz=224, isDropOut=True, numDrop=2, base_lr=0.0001)
		expIter = 52000
	#AlexNet with imgnt mean 
	elif expNum == 8:
		exp = per.alexnet_cls_pd36(nElBins=21, nAzBins=21, crpSz=224,
          meanFile='imagenet_proto')
		expIter = 26000
	#Scratch, double dropout, cropped size 224
	elif expNum == 9:
		exp = per.scratch_cls_pd36(nElBins=21, nAzBins=21, crpSz=224, 
        isDropOut=True, numDrop=2)
		expIter = 12000

	if numIter is None:
		numIter = expIter
	return exp, numIter


def make_deploy_net(exp, numIter=60000):
	imSz = exp.cPrms_.nwPrms['ipImSz']
	imSz = [[3, imSz, imSz]]
 	exp.make_deploy(dataLayerNames=['window_data'],
      newDataLayerNames=['data'], delAbove='elevation_fc',
      imSz=imSz, batchSz=256, delLayers=['slice_label'])
	modelName = exp.get_snapshot_name(numIter=numIter)
	if not osp.exists(modelName):
		print ('ModelFile %s doesnot exist' % modelName)
		return None
	net       = mp.MyNet(exp.files_['netdefDeploy'], modelName)
	#Set preprocessing
	net.set_preprocess(ipName='data', chSwap=None, noTransform=True)
	return net

##
#get images
def get_imdata(imNames, bbox, exp,  padSz=36):
	ims  = []
	imSz = exp.cPrms_.nwPrms.ipImSz 
	for imn, b in zip(imNames, bbox):
		im = cv2.imread(imn)
		x1, y1, x2, y2 = b
		h, w, ch = im.shape
		x1, y1, x2, y2, _, _ = sp3d.crop_for_imsize((h, w, x1, y1, x2, y2), 256, padSz=padSz)
		#Crop and resize
		im = cv2.resize(im[y1:y2, x1:x2, :], (imSz, imSz))
		#Mean subtaction
		if exp.cPrms_.nwPrms.meanType is None:
			#print ('MEAN SUB DONE')
			im  = im.astype(np.float32) - 128.
		else:
			raise Exception('Mean type %s not recognized' % exp.cPrms_.nwPrms.meanType)
		im = im.transpose((2,0,1))	
		ims.append(im.reshape((1,) + im.shape))
	ims = np.concatenate(ims)
	return ims

def get_predictions(exp, bench, className='car', net=None, debugMode=False):
	imNames, bbox = bench.giveTestInstances(className)	
	N = len(imNames)
	preds = []
	if net is None:
		net = make_deploy_net(exp)	
	batchSz = net.get_batchsz()	
	for i in range(0, N, batchSz):
		en  = min(i + batchSz, N)
		ims = get_imdata(imNames[i:en], bbox[i:en], exp)
		if exp.dPrms_.anglePreProc == 'classify':
			pose = net.forward(['azimuth_fc', 'elevation_fc'], **{'data':ims})
			azBin   = copy.deepcopy(pose['azimuth_fc'].squeeze())
			elBin   = copy.deepcopy(pose['elevation_fc'].squeeze())
			azBin   = np.argmax(azBin,1)
			elBin   = np.argmax(elBin,1)
			#pdb.set_trace()
			for k in range(i, en):
				az    = pep.unformat_label(azBin[k-i], None,
                exp.dPrms_, bins=exp.dPrms_.azBins)
				el    = pep.unformat_label(elBin[k-i], None, 
                exp.dPrms_, bins=exp.dPrms_.elBins)
				if debugMode:
					preds.append([(az, el, 0), (azBin[k-i], elBin[k-i], 0)])
				else:
					preds.append([az, el, 0])
	return preds


def evaluate(exp, bench, preds=None, net=None):
	if preds is None:
		preds = get_predictions(exp, bench, net=net)
	#Get the ground truth predictions
	gtPose = bench.giveTestPoses('car')
	errs   = bench.evaluatePredictions('car', preds)
	print(180*np.median(errs)/np.pi)
	return errs    			

##
#Save evaulation
def save_evaluation(exp, numIter, bench=None, forceWrite=False):
	resFile = get_result_filename(exp, numIter)	
	#Check if result file exists
	if osp.exists(resFile) and not forceWrite:
		print ('%s exists' % resFile)
		return
	#Get the benchmark object
	print ('Loading Benchmark Object')
	if bench is None:
		bench         = pbench.PoseBenchmark(classes=PASCAL_CLS)
	#Make the net
	net = make_deploy_net(exp, numIter)
	#Start evaluation
	print ('Starting evaluation')
	res = edict()
	mds = []
	for i, cls in enumerate(PASCAL_CLS):
		res[cls] = edict()
		res[cls]['pd']  = get_predictions(exp, bench, className=cls,
                     net=net) 
		res[cls]['gt']  = bench.giveTestPoses(cls)
		res[cls]['err'] = bench.evaluatePredictions(cls, res[cls]['pd']) 
		mds.append(180 * (np.median(res[cls]['err'])/np.pi))
		print ('Median accuracy on %s is %f' % (cls, mds[i]))
		res[cls]['imn'], res[cls]['bbox'] = bench.giveTestInstances(cls)
	mds = np.array(mds)
	print ('MEAN ACCURACY %f' % np.mean(mds))
	pickle.dump(res, open(resFile, 'w'))

##
#Save evaluation
def save_evaluation_multiple():
	bench         = pbench.PoseBenchmark(classes=PASCAL_CLS)
	for num in range(7):
		exp, numIter = get_exp(num)
		save_evaluation(exp, numIter, bench=bench)

##
#Evaluate for different values of iterations
def save_evaluation_multiple_iters(exp):
	bench         = pbench.PoseBenchmark(classes=PASCAL_CLS)
	numIter = range(8000,60000,4000)
	for n in numIter:
		save_evaluation(exp, n, bench=bench)


def find_theta_diff(theta1, theta2, diffType=None):
	'''
		theta1, theta2 are in radians
	'''
	v1  = np.array((math.cos(theta1), math.sin(theta1))).reshape(2,1)
	v2  = np.array((math.cos(theta2), math.sin(theta2))).reshape(2,1)
	dv  = np.sum(v1 * v2)
	if diffType is None:
		theta = math.acos(dv)
	elif diffType == 'mod180':
		theta = math.acos(np.abs(dv))
	return theta

##
def get_result_dict(expList, diffType=None):
	'''
		expList: a list of (exp, numIter)
		diffType: None - normal errors
	'''
	resMed = OrderedDict()
	resMed['expNum'] = []
	for cls in PASCAL_CLS:
		resMed[cls] = []
	resMed['mean']   = []
	for i, exps in enumerate(expList):
		resMed['expNum'].append(i)
		exp, numIter = exps
		resFile = get_result_filename(exp, numIter)
		if not osp.exists(resFile):
			save_evaluation(exp, numIter)	
		res  = pickle.load(open(resFile, 'r'))					
		md   = 0
		for cls in PASCAL_CLS:
			gtPoses = np.array(map(pep.format_radians, np.array(res[cls]['gt'])[:,0]))
			pdPoses = np.array(map(pep.format_radians, np.array(res[cls]['pd'])[:,0]))
			if diffType is None:
				err = np.median(res[cls]['err'])
			elif diffType in ['mod180']:
				diff = [find_theta_diff(th1, th2, 'mod180') for th1, th2 in zip(pdPoses,gtPoses)]
				err = np.median(diff)
			elif diffType in ['mod90']:
				err = np.median(np.mod(pdPoses - gtPoses, np.pi/2.))
			resMed[cls].append(math.degrees(err))
			md += err
		md = math.degrees(md/len(PASCAL_CLS))
		resMed['mean'].append(md)
	return resMed
	
##
#Retreive the evaluation experiments
def get_results():
	exps = []
	for num in range(7):
		exp, numIter = get_exp(num)
		exps.append([exp, numIter])
	return get_result_dict(exps)

##
#Compare results
def compare_results(diffType=None):
	numIter = 38000
	#AlexNet
	expAlex, _ = get_exp(5)
	#Ours
	expOur, _  = get_exp(2)
	#Scratch
	expScr, _  = get_exp(9)
	exps   = ([expAlex, numIter], [expOur, numIter], [expScr, numIter])
	res    = get_result_dict(exps, diffType)  		
	return res
	
##
#Debug evaluation code
def debug_evaluate_data(exp, classes=['car'], isPlot=False):
	bench         = pbench.PoseBenchmark(classes=classes)
	imNames, bbox = bench.giveTestInstances(classes[0])
	ims = get_imdata(imNames, bbox, exp) 	 	
	if isPlot:
		plt.ion()
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		for i in range(ims.shape[0]):
			im = ims[i].transpose((1,2,0))
			im = im[:,:,(2,1,0)] + 128	
			ax.imshow(im.astype(np.uint8))
			plt.show()
			plt.draw()
			ip = raw_input()
			if ip == 'q':
				return 
			plt.cla()
	return ims

def debug(exp, bench=None, net=None):
	if bench is None:
		bench = pbench.PoseBenchmark(azimuthOnly=False, classes=['car'])
	preds = get_predictions(exp, bench, net=net, debugMode=True)
	gtPose = bench.giveTestPoses('car')
	gtAz, pdAz = [], []
	gtEl, pdEl = [], []
	testPreds  = []
	for i in range(len(gtPose)):
		az, el, _ = gtPose[i]
		az,_ = pep.format_label(az, exp.dPrms_, bins=exp.dPrms_.azBins)
		el,_ = pep.format_label(el, exp.dPrms_, bins=exp.dPrms_.elBins)
		gtAz.append(az)
		gtEl.append(el)
		pdFloat, pdBins = preds[i]
		testPreds.append(pdFloat)
		paz, pel, _ = pdBins
		pdAz.append(paz)
		pdEl.append(pel)
	gtAz = np.array(gtAz)
	pdAz = np.array(pdAz)
	gtEl = np.array(gtEl)
	pdEl = np.array(pdEl)
	errs  = bench.evaluatePredictions('car', testPreds)
	print (180*np.median(errs)/np.pi)
	return np.array(gtPose), np.array(testPreds), gtAz, pdAz

def stupid_debug(exp, bench=None):
	bench = pbench.PoseBenchmark(azimuthOnly=True, classes=['car'])
	gtPose = bench.giveTestPoses('car')
	pdPose = []
	for i in range(len(gtPose)):
		a, e, _ = gtPose[i]
		aBin,_ = pep.format_label(a, exp.dPrms_, bins=exp.dPrms_.azBins)
		eBin,_ = pep.format_label(e, exp.dPrms_, bins=exp.dPrms_.elBins)
		az    = pep.unformat_label(aBin, None,
						exp.dPrms_, bins=exp.dPrms_.azBins)
		el    = pep.unformat_label(eBin, None, 
						exp.dPrms_, bins=exp.dPrms_.elBins)
		pdPose.append([az, el, 0])
	errs  = bench.evaluatePredictions('car', pdPose)
	print (np.median(errs))
	
	
