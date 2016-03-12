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


def get_result_filename(exp, numIter):
	resFile = exp.dPrms_.paths.exp.results.file
	resFile = osp.join(resFile % (osp.join(exp.cPrms_.expStr, exp.dPrms_.expStr), numIter))
	return resFile


def run_test(exp, numIter=90000, forceWrite=False, deviceId=0):
	resFile = get_result_filename(exp, numIter)
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
	data = pickle.load(open(exp.dPrms_.paths.exp.other.testData, 'r'))
	data = data['testData']
	#the parameters for extracting images
	imPrms = edict()
	imPrms['imSz']   = exp.cPrms_.nwPrms['ipImSz']
	imPrms['cropSz'] = exp.cPrms_.nwPrms['crpSz']
	imPrms['jitter_pct'] = 0
	imPrms['jitter_amt'] = 0
	imFolder = osp.join(cfg.pths.folderProc, 'imCrop', 'imSz256-align')
	gtLbs, pred    = [], []
	lbSz     = exp.dPrms_['lbPrms'].get_lbsz()
	for b in range(0, len(data), batchSz):
		print(b, 'Loading images ...')
		ims = []
		for i in range(b,b+batchSz):
			fid, imName1, imName2, gtLb = data[i]
			imName1 = osp.join(imFolder % fid, imName1)
			imName2 = osp.join(imFolder % fid, imName2)
			im = sdgr.read_double_images(imName1, imName2, imPrms)
			ims.append(im.reshape((1,) + im.shape))
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
	pred  = np.concatenate(pred)
	gtLbs = np.concatenate(gtLbs)
	pickle.dump({'pred':pred, 'gtLbs': gtLbs}, 
       open(resFile, 'w'))

def demo_test(numIter=90000):
	exp = mepg.simple_euler_dof2_dcv2_doublefcv1(gradClip=30,
        stepsize=60000, base_lr=0.001, gamma=0.1)	
	run_test(exp)


def get_rotation_performance(exp, numIter=90000):
	print ('Loading results')
	resFile = get_result_filename(exp, numIter)
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
	deltaRot, pdRot, gtRot = stv2.delta_rots(pred, gt, isOpRadian=False, opDeltaOnly=False)
	print (np.median(deltaRot), np.mean(deltaRot))	
	mdErr, counts = stv1.get_binned_angle_errs(np.array(deltaRot), np.array(gtRot))
	return mdErr, counts


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
	return exp, numIter

def eval_multiple_models(deviceId=0):
	for i in range(7,9):
		exp, numIter = get_exp(i)
		run_test(exp, numIter, forceWrite=False, deviceId=deviceId)

def get_multiple_results():
	mdErrs, counts = [], []
	#for i in range(2):
	for i in range(7,9):
		exp, numIter = get_exp(i)
		mdErr, count = get_rotation_performance(exp, numIter)
		mdErrs.append(mdErr)
		counts.append(count)
	return mdErrs, counts
