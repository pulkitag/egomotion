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
import my_pycaffe_io as mpio
import pickle
import street_config as cfg

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
def get_imdata(imNames, bbox, exp=None,  padSz=36, svMode=False):
	ims  = []
	if exp is None:
		imSz = 256
	else:
		imSz = exp.cPrms_.nwPrms.ipImSz 
	for imn, b in zip(imNames, bbox):
		if svMode:
			im = scm.imread(imn)
		else:
			im = cv2.imread(imn)
		x1, y1, x2, y2 = b
		h, w, ch = im.shape
		x1, y1, x2, y2, _, _ = sp3d.crop_for_imsize((h, w, x1, y1, x2, y2), 256, padSz=padSz)
		#Crop and resize
		im = cv2.resize(im[y1:y2, x1:x2, :], (imSz, imSz))
		if not svMode:
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

##
#Save the image data for the test set
def save_imdata():
	bench = pbench.PoseBenchmark(classes=PASCAL_CLS)
	count = 0
	dName = '/data0/pulkitag/data_sets/pascal3d/imCrop/test/im%d.jpg'
	testList = []
	count = 0
	for cls in PASCAL_CLS:
		print (cls)	
		imNames, bbox = bench.giveTestInstances(cls)	
		ims = get_imdata(imNames, bbox, svMode=True)	
		for i in range(ims.shape[0]):
			svName = dName % count
			scm.imsave(svName, ims[i])
			testList.append([imNames[i], bbox[i], svName])
			count += 1
	outFile = 'pose-files/pascal_test_data.pkl'
	pickle.dump({'testList': testList}, open(outFile, 'w'))


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


def get_data_dict(setName):
	if setName == 'test':
		fName = './pose-files/pascal3d_dict_test_imSz256_pdSz36.pkl'
	else:
		fName = './pose-files/pascal3d_dict_train_imSz256_pdSz24.pkl'
	dat   = pickle.load(open(fName, 'r'))
	dat   = dat['fStore']
	return dat

def transform_dict(setName):
	dat = get_data_dict(setName)
	fStore = {}
	for k in dat.keys():
		inData = dat[k]
		for i, n in enumerate(inData.name):
			ns = n.split('/')
			key = '%s/%s' % (ns[1],ns[2])
			fStore[key]	= [k, inData.lbs[i]]
	return fStore

def load_single_feature(featFile, netName='caffe_pose_fc5'):
	dat      = pickle.load(open(featFile, 'r'))
	return dat[netName]

def load_train_features(keyList, netName='caffe_pose_fc5', imSz=256, padSz=36):
	dirName = '/data0/pulkitag/nn/pascal3d_imSz256_pad36_hash_features_08mar16/imSz256_pad36_hash'
	feats = []	
	for k in keyList:
		featFile = osp.join(dirName, k[0:-4] + '.p')
		dat      = pickle.load(open(featFile, 'r'))
		feats.append(dat[netName])
	feats = np.concatenate(feats)
	return feats

def load_test_features(netName='caffe_pose_fc5'):
	outFile = './pose-files/pascal_test_data.pkl'
	dat     = pickle.load(open(outFile, 'r'))
	dat     = dat['testList']
	dirName = '/data0/pulkitag/nn/pascal3d_test2_features_08mar16'
	feats   = []
	otherDat = []
	for tt in dat:
		origName, bbox, svName = tt
		featName = osp.basename(svName[0:-4] + '.p')
		featName = osp.join(dirName, featName)	
		feats.append(load_single_feature(featName, netName))	
		otherDat.append([origName, bbox])
	feats = np.concatenate(feats)
	return feats, otherDat


def compute_accuracy_nn(netName='caffe_pose_fc5'):
	trainDat   = transform_dict('train')
	keyList    = trainDat.keys()
	trainFeats = load_train_features(keyList, netName)
	testFeats, metaDat  = load_test_features(netName) 
	#Find the neartest neigbhors
	nnIdxs     = find_nn(testFeats, trainFeats)
	nnKeys     = []
	for i,_ in enumerate(nnIdxs):
		idx = []
		for k in nnIdxs[i]:
			idx.append(keyList[k])
		nnKeys.append(idx)
	pickle.dump({'testInfo':metaDat, 'nnKeys': nnKeys}, open('pascal_results_%s.pkl' % netName,'w'))


def match_bbox(b1, b2):
	a = True
	for i in range(4):
		a = a and (b1[i]==b2[i])
	return a

def eval_accuracy_nn(bench=None, netName='caffe_pose_fc5', classes=['car'], visMatches=False):
	modErr = []	
	if bench is None:
		bench = pbench.PoseBenchmark(classes=classes)
	#Train data
	trainDat   = transform_dict('train')
	keyList    = trainDat.keys()
	#result data
	resDat    = pickle.load(open('pascal_results/pascal_results_%s.pkl' % netName,'r'))
	resImList  = [l[0] for l in resDat['testInfo']]
	resBBox    = [l[1] for l in resDat['testInfo']] 
	resKeys    = resDat['nnKeys'] 
	imNames, bbox = bench.giveTestInstances(classes[0])
	gtPoses       = bench.giveTestPoses(classes[0])
	preds = []
	if visMatches:
		plt.ion()
		fig = plt.figure()
		ax  = []
		count = 1
		for i in range(3):
			for j in range(2):
				ax.append(fig.add_subplot(2,3, count))
				count += 1
	
	exampleCount = 0	
	for nm, bb in zip(imNames, bbox):
		#nm is the name of the image for which we want to find the pose
		idx = [i for i,l in enumerate(resImList) if l == nm]
		if len(idx) > 1:
			for dd in idx:
				resBox  = resBBox[dd]
				isFound = match_bbox(resBox, bb)
				if isFound:
					idx = [dd]
					break
			if not isFound:
				pdb.set_trace() 
		assert len(idx)==1
		idx = idx[0]
		#The 1-NN
		if visMatches:
			dirName  = osp.join(cfg.pths.pascal.dataDr, 'imCrop',
           'imSz256_pad36_hash', 'imSz256_pad36_hash')
			nnImNames = resKeys[idx]
			ax[0].imshow(get_imdata([nm], [bb], svMode=True)[0])
			for vv, visname in enumerate(nnImNames[0:5]):
				im = scm.imread(osp.join(dirName, visname))
				ax[vv+1].imshow(im)
			plt.show()
			plt.draw()
			ip = raw_input()
			if ip =='q':
				return
		key = resKeys[idx][0]
		_, pred = trainDat[key]	
		#print (gtPoses[exampleCount])	
		modErr.append(find_theta_diff(pred[0], gtPoses[exampleCount][0], 'mod180'))
		exampleCount += 1
		pred = pred + (0.,)
		preds.append(pred)
	errs  = bench.evaluatePredictions(classes[0], preds)
	modErr = np.array(modErr)
	mdModErr = 180 * np.median(modErr)/np.pi
	mdErr    = 180*np.median(errs)/np.pi
	return mdModErr, mdErr


def save_nn_results_final(netName='caffe_pose_fc5'):
	res = edict()
	for cls in PASCAL_CLS:
		res[cls] = eval_accuracy_nn(classes=[cls],netName=netName)
	return res	
			
	

def find_nn(feats1, feats2):
	idxs = [] 
	for i1 in range(feats1.shape[0]):
		f1   = feats1[i1]
		diff = feats2 - f1
		diff = np.sum(diff * diff,1)
		sortIdx = np.argsort(diff)
		idxs.append(sortIdx[0:10])
	return idxs

def save_nn_results(cls='car', bench=None):
	dat = get_data_dict('train')
	fullKeys = dat.keys()
	idKeys   = [osp.split(k)[1] for k in dat.keys()]
	trainFiles, trainLbs = get_cls_set_files('train', cls=cls)
	if bench is None:
		bench = pbench.PoseBenchmark(classes=[cls])
	imNames, bbox = bench.giveTestInstances(cls)	
	dat	

def verify_dict():
	data = transform_dict()
	fig = plt.figure()
	for k in data.keys():
		imName = '/data0/pulkitag/data_sets/pascal3d/imCrop/imSz256_pad36_hash/imSz256_pad36_hash'
		imName = osp.join(imName, k)
		im = scm.imread(imName)		
		plt.imshow(im)
		plt.ion()
		plt.show()
		plt.title('az: %f, el:%f' % data[k][1])
		ip = raw_input()
		if ip == 'q':
			return
	
def get_cls_set_files(setName='train', cls='car'):
	imSz, padSz = 256, 36
	inFile = 'pose-files/euler_%s_pascal3d_imSz%d_pdSz%d.txt' % (setName, imSz, padSz)
	fid    = mpio.GenericWindowReader(inFile)
	fNames, lbs = [], []
	while True:
		if fid.is_eof():
			break
		imDat, lbDat = fid.read_next()
		fName, ch, h, w, x1, y1, x2, y2 = imDat[0].strip().split()
		if cls in fName:
			fNames.append(fName)
			lbs.append(lbDat[0])
	fid.close()
	return fNames, lbs


def find_test_keys(cls='car', bench=None, dat=None):
	if dat is None:
		dat = get_data_dict('test')
	fullKeys = dat.keys()
	idKeys   = [osp.split(k)[1] for k in fullKeys]
	testKeys = []
	if bench is None:
		bench = pbench.PoseBenchmark(classes=[cls])
	imNames, bbox = bench.giveTestInstances(cls)	
	for nm in imNames:
		_, pascalid = osp.split(nm)
		pascalid = pascalid[0:-4]
		idx = idKeys.index(pascalid)
		if len(dat[fullKeys[idx]].coords) > 1:
			pdb.set_trace()
		testKeys.append(fullKeys[idx])			
	return testKeys


	

