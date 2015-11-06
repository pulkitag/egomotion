import numpy as np
import my_pycaffe_io as mpio
import street_utils as su
import my_exp_v2 as mev2
import street_exp as se
from easydict import EasyDict as edict
from os import path as osp

def create_window_file():
	setName = ['test', 'train']
	for i,s in enumerate(setName):
		inName = 'pose-files/annotations_master_%s_pascal3d.txt' % s
		oName  = 'pose-files/euler_%s_pascal3d.txt' % s
		inFid  = mpio.GenericWindowReader(inName)
		imDat, lbls = [], []
		N = inFid.num_
		for i in range(inFid.num_):
			im, lb = inFid.read_next()
			imDat.append(im)
			lbls.append(lb)
		inFid.close()
		randSeed = 3 + (2 * i)
		randState = np.random.RandomState(randSeed)
		perm = randState.permutation(N)

		if s == 'train':
			numBad = 2
		else:
			numBad = 0
		oFid = mpio.GenericWindowWriter(oName, N-numBad, 1, 3)
		for p in perm:
			im, lb = imDat[p], lbls[p]
			fName, ch, h, w, x1, y1, x2, y2 = im[0].strip().split()
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			if x2 <= x1 or y2 <= y1:
				print ('Size is weird', x1,x2,y1,y2)
				print ('Skipping', s, im)
				continue
			if x1 <0 or y1<0:
				print ('Too small', x1, y1)
			if x2 > w or y2 > h:
				print ('Too big', x2, w, y2, h)	
			rots = []
			for theta in lb[0]:
				rots.append(su.rot_range(theta)/30.0)
			rots.append(1.0)
			oFid.write(rots, *im)
		oFid.close()
	

def setup_experiment(isRun=False):
	#Get my best multiloss net
	prms, cPrms = mev2.ptch_pose_euler_smallnet_v5_fc5_exp1_lossl1()
	#lrPrms      = se.get_lr_prms()
	lrPrms       = cPrms.lrPrms
	finePrms    = edict() 
	
	codeDir = '/work4/pulkitag-code/code/projStreetView'	
	finePrms.isSiamese = False
	finePrms['solver'] = lrPrms['solver'] 
	finePrms.paths = edict()
	finePrms.paths.imRootDir  = '/data0/pulkitag/data_sets/pascal_3d-my-copy/PASCAL3D+_release1.1/Images/'
	finePrms.paths.windowFile = edict()
	finePrms.paths.windowFile.train = osp.join(codeDir, 'pose-files/euler_train_pascal3d.txt')
	finePrms.paths.windowFile.test  = osp.join(codeDir, 'pose-files/euler_test_pascal3d.txt')
	#How many layers to finetune
	finePrms.lrAbove = None
	#Jittering
	finePrms.jitter_pct = 0.1
	finePrms.jitter_amt = 0
	#Name of the experiment
	finePrms.expName = 'pascal3d_euler_%s' % lrPrms.expStr 
	exp,modelFile = se.setup_experiment_for_finetune(prms, cPrms, finePrms, 60000) 
	exp.del_layer('ptch_loss')
	exp.del_layer('accuracy')
	exp.del_layer('ptch_fc')
	if isRun:
		exp.make(modelFile=modelFile)
		exp.run()
	
	return exp	
	
