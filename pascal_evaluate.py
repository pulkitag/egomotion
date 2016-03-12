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

PASCAL_CLS = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
              'chair', 'diningtable', 'motorbike', 'sofa', 'train',
              'tvmomitor']
def get_exp():
	pass

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
#Get the objec that would be used to generate the benchmark data
def get_bench_obj():
	bench = pbench.PoseBenchmark('car')
	return bench

##
#get images
def get_imdata(imNames, bbox, exp,  padSz=24):
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


def get_predictions(exp, bench, net=None, debugMode=False):
	imNames, bbox = bench.giveTestInstances('car')	
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


def get_evaluate_data(exp, classes=['car'], isPlot=False):
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

def evaluate(exp, bench, preds=None, net=None):
	if preds is None:
		preds = get_predictions(exp, bench, net=net)
	#Get the ground truth predictions
	gtPose = bench.giveTestPoses('car')
	errs   = bench.evaluatePredictions('car', preds)
	print(np.median(errs))
	return errs    			


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
	print (np.median(errs))
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
	
	
