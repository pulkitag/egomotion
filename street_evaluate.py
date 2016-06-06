import street_exp_v2 as sev2
import street_process_data as spd
import street_config as cfg
import pickle
import numpy as np
import street_label_utils as slu
import my_exp_pose_grps as mepg
import my_pycaffe as mp
from os import path as osp
from easydict import EasyDict as edict
import streetview_data_group_rots as sdgr
import street_test_v2 as stv2
import street_test as stv1
import copy
import other_utils as ou
import math

REAL_PATH = cfg.REAL_PATH

#First lets make a proper test set :) 
def make_test_set(dPrms, numTest=100000):
	listName = dPrms['paths'].exp.other.grpList % 'test'
	data     = pickle.load(open(listName, 'r'))	
	grpDat   = []
	grpCount = []
	numGrp   = 0
	for i,g in enumerate(data['grpFiles']):
		grpDat.append(pickle.load(open(g, 'r'))['groups'])
		grpCount.append(len(grpDat[i]))
		print ('Groups in %s: %d' % (g, grpCount[i]))
		numGrp += grpCount[i]
	print ('Total number of groups: %d' % numGrp)
	grpSampleProb = [float(i)/float(numGrp) for i in grpCount]
	randSeed  = 7
	randState = np.random.RandomState(randSeed) 
	elms      = []
	for t in range(numTest):
		if np.mod(t,5000)==1:
			print(t)	
		breakFlag = False
		while not breakFlag:
			rand   =  randState.multinomial(1, grpSampleProb)
			grpIdx =  np.where(rand==1)[0][0]
			ng     =  randState.randint(low=0, high=grpCount[grpIdx])
			grp    =  grpDat[grpIdx][ng]
			l1     =  randState.permutation(grp.num)[0]
			l2     =  randState.permutation(grp.num)[0]
			if l1==l2:
				rd = randState.rand()
				#Sample the same image rarely
				if rd < 0.85:
					continue
			elm = [grp.folderId, grp.crpImNames[l1], grp.crpImNames[l2]]
			lb  = slu.get_pose_delta(dPrms['lbPrms'].lb, grp.data[l1].rots,
            grp.data[l2].rots, grp.data[l1].pts.camera,
            grp.data[l2].pts.camera)
			lb  = np.array(lb)
			elm.append(lb)
			elms.append(elm)
			breakFlag = True
	pickle.dump({'testData': elms}, open(dPrms.paths.exp.other.testData, 'w'))
	return elms

#Test set that amir originally used.
def get_test_set_amir():
	fName    = osp.join(cfg.pths.data0, 'data_sets', 'streetview', 'test',
             'regression_data', '30_sfrtest_uni.txt')
	imFolder = osp.dirname(fName)
	folderId = 'regression_test'
	fid      = open(fName, 'r')
	lines    = fid.readlines()
	fid.close()
	elms     = []	
	for l in lines:
		fName1, fName2, az, el = l.split()
		az = float(az)
		el = float(el)
		lb = np.array([math.radians(az), math.radians(el)])
		elm   = [folderId, fName1, fName2] 
		elm.append(lb)
		elms.append(elm)	
	return elms, imFolder

#eval amir on amir.
def eval_amir_on_amir():
	fName    = osp.join(cfg.pths.data0, 'data_sets', 'streetview', 'test',
             'regression_data', 'regression_test_eval.txt')
	fid      = open(fName, 'r')
	lines    = fid.readlines()
	N        = len(lines)
	fid.close()
	gt = np.zeros((N,2))
	pd = np.zeros((N,2))
	for i, l in enumerate(lines):
		fName1, fName2, azGt, elGt, az, el = l.split()
		azGt, az = float(azGt), float(az)
		elGt, el = float(elGt), float(el)
		azGt, az = math.radians(azGt), math.radians(az)
		elGt, el = math.radians(elGt), math.radians(el)
		gt[i,:]  = azGt, elGt
		pd[i,:]  = az, el
	return compute_rotation_errors(pd, gt)


def save_amir_set_preds(exp, numIter):
	resFile = get_result_filename(exp, numIter, amirTest=True)
	data = pickle.load(open(resFile, 'r'))
	pred = data['pred']
	gtLb = data['gtLbs']
	outName = osp.join('./test-files/amir_test',
            exp.cPrms_.expStr, exp.dPrms_.expStr + '-numIter%d.txt')
	outName = outName % numIter
	ou.mkdir(osp.dirname(outName))
	elms, _ = get_test_set_amir()
	#open the file
	fid = open(outName, 'w')
	allPd = np.zeros((len(pred),2))
	allGt = np.zeros((len(pred),2))
	count = 0
	for (el, pd, gt) in zip(elms, pred, gtLb):
		_, nm1, nm2,_ = el
		allPd[count,:] = pd[[1,0]]
		allGt[count,:] = gt[[1,0]]
		azGt, elGt = math.degrees(gt[0]), math.degrees(gt[1])
		azPd, elPd = math.degrees(pd[0]), math.degrees(pd[1])
		l = '%s %s %f %f' % (nm1, nm2, azGt, elGt)
		l = '%s %f %f %f %f %f %f' % (l, azPd, elPd, 0, pd[2], pd[3], pd[4])
		fid.write('%s\n' % l)
		count += 1
	fid.close()
	mdErr, _ = compute_rotation_errors(allPd, allGt) 	
	print mdErr
		
def make_test_set_amir():
	elms, imFolder = get_test_set_amir()
	pickle.dump({'testData': elms}, open('./test-files/test_regress_amir.pkl', 'w'))


def vis_test_set_amir():
	### THIS iS NOT COMPLETE ####
	fName    = osp.join(cfg.pths.data0, 'data_sets', 'streetview', 'test',
             'regression_data', '30_sfrtest_uni.txt')
	folderId = 'regression_test'
	fid      = open(fName, 'r')
	lines    = fid.readlines()
	fid.close()

def demo_make_test():
	posePrms = slu.PosePrms(maxRot=90, simpleRot=True, dof=2)
	dPrms   =  sev2.get_data_prms(lbPrms=posePrms)
	make_test_set(dPrms)


def make_deploy_net(exp, numIter=60000, deviceId=0):
	imSz = exp.cPrms_.nwPrms['ipImSz']
	imSz = [[6, imSz, imSz]]
 	exp.make_deploy(dataLayerNames=['window_data'],
      newDataLayerNames=['pair_data'], delAbove='pose_fc',
      imSz=imSz, batchSz=100)
	modelName = exp.get_snapshot_name(numIter=numIter)
	if not osp.exists(modelName):
		print ('ModelFile %s doesnot exist' % modelName)
		return None
	net       = mp.MyNet(exp.files_['netdefDeploy'], modelName,
              deviceId=deviceId)
	#Set preprocessing
	net.set_preprocess(ipName='pair_data', chSwap=None, noTransform=True)
	return net


def get_result_filename(exp, numIter, amirTest=False):
	if amirTest:
		eStr = 'amir_test'
	else:
		eStr = ''
	resFile = exp.dPrms_.paths.exp.results.file
	resFile = osp.join(resFile % (osp.join(exp.cPrms_.expStr, exp.dPrms_.expStr + eStr), 
            numIter))
	return resFile


def run_test(exp, numIter=90000, forceWrite=False, deviceId=0, amirTest=False):
	resFile = get_result_filename(exp, numIter, amirTest)
	dirName = osp.dirname(resFile)
	ou.mkdir(dirName)
	if osp.exists(resFile) and not forceWrite:
		print ('Result file for %s exists' % resFile)
		return
	net = make_deploy_net(exp, numIter=numIter, deviceId=deviceId)
	if net is None:
		return
	batchSz = net.get_batchsz()
	#Load the test data
	if amirTest:
		fName     = './test-files/test_regress_amir.pkl'
		imFolder  = osp.join(cfg.pths.data0, 'data_sets', 'streetview', 'test',
             'regression_data', '%s')
	else:
		fName = exp.dPrms_.paths.exp.other.testData
		imFolder = osp.join(cfg.pths.folderProc, 'imCrop', 'imSz256-align')
	data = pickle.load(open(fName, 'r'))
	data = data['testData']
	#the parameters for extracting images
	imPrms = edict()
	imPrms['imSz']   = exp.cPrms_.nwPrms['ipImSz']
	imPrms['cropSz'] = exp.cPrms_.nwPrms['crpSz']
	imPrms['jitter_pct'] = 0
	imPrms['jitter_amt'] = 0
	gtLbs, pred    = [], []
	lbSz     = exp.dPrms_['lbPrms'].get_lbsz()
	for b in range(0, len(data), batchSz):
		print(b, 'Loading images ...')
		ims = []
		for i in range(b,min(len(data), b+batchSz)):
			fid, imName1, imName2, gtLb = data[i]
			imName1 = osp.join(imFolder % fid, imName1)
			imName2 = osp.join(imFolder % fid, imName2)
			im = sdgr.read_double_images(imName1, imName2, imPrms)
			ims.append(im.reshape((1,) + im.shape))
			if amirTest:
				gtLbs.append(np.array(gtLb).reshape((1,2)))	
			else:
				gtLbs.append(np.array(gtLb).reshape((1,lbSz)))	
		ims  = np.concatenate(ims, axis=0)
		#WHERE IS THE MEAN SUBTRACTION .... GRRRRRRR
		if exp.cPrms_.nwPrms.meanType is None:
			print ('MEAN SUB DONE')
			ims  = ims.astype(np.float32) - 128.
		else:
			raise Exception('Mean type %s not recognized' % exp.cPrms_.nwPrms.meanType)	
		pose = net.forward(['pose_fc'], **{'pair_data':ims})
		pred.append(copy.deepcopy(pose['pose_fc']))
	gtLbs = np.concatenate(gtLbs)
	pred  = np.concatenate(pred)
	print (pred.shape, gtLbs.shape)
	#See if the labels require normalization	
	if exp.dPrms_['lbPrms'].lb.nrmlz is not None:
		nrmlzDat = pickle.load(open(exp.dPrms_.paths.exp.other.poseStats, 'r'))
		lbInfo   = copy.deepcopy(exp.dPrms_['lbPrms'].lb)
		lbInfo['nrmlzDat'] = nrmlzDat
		for i in range(pred.shape[0]):
			pred[i] = slu.unnormalize_label(pred[i], lbInfo['nrmlzDat'])	
	pickle.dump({'pred':pred, 'gtLbs': gtLbs}, 
       open(resFile, 'w'))


def demo_test(numIter=90000):
	exp = mepg.simple_euler_dof2_dcv2_doublefcv1(gradClip=30,
        stepsize=60000, base_lr=0.001, gamma=0.1)	
	run_test(exp)


def compute_rotation_errors(pred, gt):
	'''
		pred, gt should be in pitch, yaw, roll format
	'''	
	deltaRot, pdRot, gtRot = stv2.delta_rots(pred, gt, isOpRadian=False, opDeltaOnly=False)
	print (np.median(deltaRot), np.mean(deltaRot))	
	mdErr, counts = stv1.get_binned_angle_errs(np.array(deltaRot), np.array(gtRot))
	return mdErr, counts


def get_rotation_performance(exp, numIter=90000, amirTest=False):
	print ('Loading results')
	resFile = get_result_filename(exp, numIter, amirTest=amirTest)
	data = pickle.load(open(resFile, 'r'))
	pred = data['pred']
	gt   = data['gtLbs']
	#Convert from yaw, pitch, roll to pitch, yaw, roll
	if exp.dPrms_.lbPrms.lb.dof in [2,5]:
		pred = pred[:,[1,0]]
		gt   = gt[:,[1,0]]
	else:
		pred = pred[:,[1,0,2]]
		gt   = gt[:,[1,0,2]]
	return compute_rotation_errors(pred, gt)


def get_translation_peformance(exp, numIter=90000):
	print ('Loading results')
	resFile = get_result_filename(exp, numIter)
	data = pickle.load(open(resFile, 'r'))
	pred = data['pred']
	gt   = data['gtLbs']
	if exp.dPrms_.lbPrms.lb.dof == 5:
		pred = pred[:, 2:]
		gt   = gt[:,2:]
	elif exp.dPrms_.lbPrms.lb.dof == 6:
		pred = pred[:,3:]
		gt   = gt[:,3:]
	err = np.abs(pred - gt)
	return gt, pred, err		
	

def plot_peformance():
	amirErr =  [10.6, 13.9, 17.62, 18.2, 22.4, 23.5, 27.8]
	myErr   =  [3.7, 6.07, 9.39, 13.16, 19.10, 24.5, 35.8]
	bins    =  [4.5, 18.9, 37.8, 56.8,  75.8,  94.8, 113.8]   
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	l1, = ax.plot(bins, myErr, 'r', linewidth=3)
	l2, = ax.plot(bins, amirErr, 'b', linewidth=3)
	plt.legend([l1, l2], ['caffenet', 'torchnet'])
	plt.title('Pose Performance')
	plt.savefig('pose_performance.pdf')
	

def get_exp(expNum):
	if expNum == 0:
		exp = mepg.simple_euler_dof2_dcv2_doublefcv1(gradClip=30, stepsize=60000,
			 base_lr=0.001, gamma=0.1)
		numIter = 182000
		#numIter = 130000
		#numIter = 84000
	elif expNum == 1:
		exp = mepg.simple_euler_dof2_dcv2_doublefcv1(gradClip=30, stepsize=60000, 
			 base_lr=0.0001, gamma=0.5)
		numIter = 182000
		#numIter = 130000
		#numIter = 84000
	elif expNum == 2:
		exp = mepg.simple_euler_dof2_dcv2_smallnetv5(gradClip=30, stepsize=60000,
			 gamma=0.5, base_lr=0.0001) 	
		numIter = 84000
	elif expNum == 3:
		exp = mepg.simple_euler_dof2_dcv2_smallnetv5(gradClip=30, stepsize=60000)
		numIter = 84000
	elif expNum == 4:
		exp = mepg.simple_euler_dof2_dcv2_smallnetv5(gradClip=30)
		numIter = 82000
	elif expNum==5:
		exp = mepg.simple_euler_dof2_dcv2_doublefcv1_diff(gradClip=30,
        stepsize=60000, base_lr=0.001, gamma=0.1)
		numIter = 84000
	#Using diff concat instead of just concatenating
	elif expNum == 6:
		exp = mepg.simple_euler_dof2_dcv2_doublefcv1_diff(gradClip=30,
        stepsize=60000, base_lr=0.001, gamma=0.1)
		numIter = 182000
	#5DOF experiment
	elif expNum == 7:
		exp = mepg.simple_euler_dof5_dcv2_doublefcv1(gradClip=30,
           stepsize=60000, gamma=0.1, base_lr=0.0003)
		numIter = 82000
	#5DOF experiment, different learning rate
	elif expNum == 8:
		exp = mepg.simple_euler_dof5_dcv2_doublefcv1(gradClip=30,
           stepsize=60000, gamma=0.1, base_lr=0.001)
		numIter = 82000
	#5DOF experiment, with l2 loss
	elif expNum == 9:
		exp = mepg.simple_euler_dof5_dcv2_doublefcv1_l2loss(gradClip=30,
           stepsize=60000, gamma=0.1, base_lr=0.001)
		numIter = 82000
	#5DOF experiment, with l2 loss but 84000 iterations - to measure noise level
	elif expNum == 10:
		exp = mepg.simple_euler_dof5_dcv2_doublefcv1_l2loss(gradClip=30,
           stepsize=60000, gamma=0.1, base_lr=0.001)
		numIter = 84000
	#5DOF experiment, with l2 loss but 80000 iterations - to measure noise level
	elif expNum == 11:
		exp = mepg.simple_euler_dof5_dcv2_doublefcv1_l2loss(gradClip=30,
           stepsize=60000, gamma=0.1, base_lr=0.001)
		numIter = 80000

	return exp, numIter

def eval_multiple_models(deviceId=0, forceWrite=False):
	for i in range(10,12):
		print (i)
		exp, numIter = get_exp(i)
		run_test(exp, numIter, forceWrite=forceWrite, deviceId=deviceId)

def get_multiple_results():
	mdErrs, counts = [], []
	#for i in range(2):
	for i in range(9,12):
		exp, numIter = get_exp(i)
		mdErr, count = get_rotation_performance(exp, numIter)
		mdErrs.append(mdErr)
		counts.append(count)
	return mdErrs, counts
