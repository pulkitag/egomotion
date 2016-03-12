import pascal_exp as pep
import my_exp_config as mec
import street_config as cfg
from os import path as osp
import my_exp_pose_grps as mepg

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

def scratch_cls_pd36(isRun=False, nAzBins=18, nElBins=18,
                  isLog=True, gradClip=30, stepsize=10000,
                  gamma=0.5, base_lr=0.001, deviceId=0,
                 isDropOut=False, numDrop=1):
	dPrms   = pep.get_data_prms(anglePreProc='classify', 
             nAzBins=nAzBins, nElBins=nElBins)
	nwFn    = pep.process_net_prms
	ncpu = 0
	if isDropOut:
		if numDrop == 1:
			baseProto = 'doublefc-v1_window_fc6_dropout'
		else:
			baseProto = 'doublefc-v1_window_fc6_double_dropout'
	else:
		baseProto = 'doublefc-v1_window_fc6'

	nwArgs  = {'ncpu': ncpu, 'lrAbove': None, 'preTrainNet':None,
             'dataNetDefProto': 'data_layer_pascal_cls',
						 'baseNetDefProtp': baseProto,
             'lossNetDefProto': 'pascal_pose_loss_classify_layers'}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': gradClip,
            ' stepsize': stepsize, 'base_lr': base_lr, 'gamma': gamma}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn = pep.make_net_def, isLog=isLog)
	if isRun:
		exp.make(deviceId=deviceId)
		exp.run() 
	return exp 	 			


def alexnet_cls_pd36(isRun=False, nAzBins=18, nElBins=18,
                  isLog=True, gradClip=30, stepsize=10000,
                  gamma=0.5, base_lr=0.001, deviceId=0, crpSz=240,
									lrAbove=None, meanFile='None'):
	dPrms   = pep.get_data_prms(anglePreProc='classify', 
             nAzBins=nAzBins, nElBins=nElBins)
	nwFn    = pep.process_net_prms
	ncpu = 0
	preTrainNet = osp.join(cfg.pths.data0,\
              'caffe_models/bvlc_reference/bvlc_reference_caffenet_upgraded.caffemodel')
	if not meanFile == 'None':
		meanFile    = osp.join(cfgs.pths.data0,\
								'caffe_models/ilsvrc2012_mean.binaryproto')
	nwArgs  = {'ncpu': ncpu, 'lrAbove': lrAbove, 'preTrainNet':preTrainNet,
             'dataNetDefProto' : 'data_layer_pascal_cls',
             'lossNetDefProto' : 'pascal_pose_loss_classify_layers',
						 'baseNetDefProto' : 'alexnet',
						 'ipImSz': 227, 'crpSz': crpSz,
						 'opLrMult': 10, meanFile:meanFile}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': gradClip,
            ' stepsize': stepsize, 'base_lr': base_lr, 'gamma': gamma}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn = pep.make_net_def, isLog=isLog)
	if isRun:
		exp.make(deviceId=deviceId)
		exp.run() 
	return exp 	 			


def torchnet_cls_pd36(isRun=False, nAzBins=18, nElBins=18,
                  isLog=True, gradClip=30, stepsize=10000,
                  gamma=0.5, base_lr=0.001, deviceId=0):
	dPrms   = pep.get_data_prms(anglePreProc='classify', 
             nAzBins=nAzBins, nElBins=nElBins)
	nwFn    = pep.process_net_prms
	ncpu = 0
	preTrainNet = osp.join(cfg.pths.data0,\
                'caffe_models/streetview/pose-l1-torch7.caffemodel')
	nwArgs  = {'ncpu': ncpu, 'lrAbove': None, 'preTrainNet':preTrainNet,
             'dataNetDefProto' : 'data_layer_pascal_cls',
             'lossNetDefProto' : 'pascal_pose_loss_classify_layers',
						 'baseNetDefProto' : 'pose-l1-torch7',
						 'ipImSz': 101, 
						 'opLrMult': 10}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': gradClip,
            ' stepsize': stepsize, 'base_lr': base_lr, 'gamma': gamma}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn = pep.make_net_def, isLog=isLog)
	if isRun:
		exp.make(deviceId=deviceId)
		exp.run() 
	return exp 	 			


def doublefcv1_dcv2_dof2net_cls_pd36(isRun=False, nAzBins=18, nElBins=18,
                  isLog=True, gradClip=30, stepsize=10000,
                  gamma=0.5, base_lr=0.001, deviceId=0, crpSz=240,
                  lrAbove=None, isDropOut=False, numDrop=1):
	'''
		isDropout: if dropouts should be used
		numDrop  : number of dropout layers
	'''
	#Source net
	srcExp   = mepg.simple_euler_dof2_dcv2_doublefcv1(gradClip=30,
            stepsize=60000, base_lr=0.001, gamma=0.1)
	srcIter  = 182000
	preTrainNet  = srcExp.get_snapshot_name(srcIter)
	#PASCAL Settings 
	dPrms   = pep.get_data_prms(anglePreProc='classify', 
             nAzBins=nAzBins, nElBins=nElBins)
	nwFn    = pep.process_net_prms
	ncpu = 0
	if isDropOut:
		if numDrop == 1:
			baseProto = 'doublefc-v1_window_fc6_dropout'
		else:
			baseProto = 'doublefc-v1_window_fc6_double_dropout'
	else:
		baseProto = 'doublefc-v1_window_fc6'
	nwArgs  = {'ncpu': ncpu, 'lrAbove': lrAbove, 'preTrainNet':preTrainNet,
             'dataNetDefProto' : 'data_layer_pascal_cls',
             'lossNetDefProto' : 'pascal_pose_loss_classify_layers',
						 'baseNetDefProto' : baseProto,
						 'ipImSz': 101, 'crpSz': crpSz,
						 'opLrMult': 10}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': gradClip,
            ' stepsize': stepsize, 'base_lr': base_lr, 'gamma': gamma}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn = pep.make_net_def, isLog=isLog)
	if isRun:
		exp.make(deviceId=deviceId)
		exp.run() 
	return exp 	 			


