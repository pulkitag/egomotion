import my_exp_config as mec
import os
from os import path as osp
import socket
from easydict import EasyDict as edict
import other_utils as ou
import my_pycaffe_utils as mpu
import street_config as cfg
import street_label_utils as slu
import street_process_data as spd
import pickle
import copy
import numpy as np
import street_exp_v2 as sev2

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

##
#get the paths
def get_paths(dPrms):
	if dPrms is None:
		dPrms = data_prms()
	expDir, dataDir = cfg.pths.pascal.expDr, cfg.pths.pascal.dataDr
	ou.mkdir(expDir)
	pth        = edict()
	#All the experiment paths
	pth.exp    = edict() 
	pth.exp.dr = expDir
	#Snapshots
	pth.exp.snapshot    = edict()
	pth.exp.snapshot.dr = osp.join(pth.exp.dr, 'snapshot')
	ou.mkdir(pth.exp.snapshot.dr)
	pth.exp.results = edict()
	#Results
	pth.exp.results.dr   = osp.join(pth.exp.dr, 'results', '%s')
	pth.exp.results.file = osp.join(pth.exp.results.dr, 'iter%d.pkl') 
	#Data files
	pth.data      = edict()
	pth.data.dr   = dataDir	
	pth.data.imFolder = osp.join(pth.data.dr, 'imCrop', 'imSz%d_pad%d')
	pth.data.imFolder = pth.data.imFolder % (dPrms.imCutSz, dPrms.imPadSz)
	#base net files
	pth.baseProto = osp.join(REAL_PATH, 'base_files', '%s.prototxt')
	#Window files
	windowDr      = osp.join(REAL_PATH, 'pose-files')
	pth.window  = edict()
	pth.window.train = osp.join(windowDr, 'euler_train_pascal3d_imSz%d_pdSz%d.txt')
	pth.window.test  = osp.join(windowDr, 'euler_test_pascal3d_imSz%d_pdSz%d.txt')
	pth.window.train = pth.window.train % (dPrms.imCutSz, dPrms.imPadSz)
	pth.window.test  = pth.window.test %  (dPrms.imCutSz, dPrms.imPadSz)
	return pth	


##
#Parameters that govern what data is being used
def get_data_prms(dbFile=DEF_DB % 'pascal_data', **kwargs):
	dArgs   = edict()
	dArgs.dataset = 'pascal'
	dArgs.imCutSz = 256
	dArgs.imPadSz = 36
	allKeys = dArgs.keys()  
	dArgs   = mpu.get_defaults(kwargs, dArgs)	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
	dArgs['paths']  = get_paths(dArgs) 
	return dArgs

##
#Parameters that govern the structure of the net
def net_prms(dbFile=DEF_DB % 'pascal_net', **kwargs):
	dArgs = mec.get_default_net_prms(dbFile, **kwargs)
	del dArgs['expStr']
	#The data NetDefProto
	dArgs.dataNetDefProto = 'data_layer_pascal_reg' 
	#the basic network architecture: baseNetDefProto
	dArgs.baseNetDefProto = 'doublefc-v1_window_fc6'
	#the loss layers:
	dArgs.lossNetDefProto = 'pascal_pose_loss_log_l1_layers'
	if dArgs.batchSize is None:
		dArgs.batchSize = 128 
	#The amount of jitter in both the images
	dArgs.maxJitter = 0
	#The size of crop that should be cropped from the image
	dArgs.crpSz     = 227
	#the size to which the cropped image should be resized
	dArgs.ipImSz    = 101
	##The mean file
	dArgs.meanFile  = ''
	dArgs.meanType  = None
	dArgs.ncpu      = 3
	dArgs   = mpu.get_defaults(kwargs, dArgs, False)
	allKeys = dArgs.keys()	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs, ignoreKeys=['ncpu'])
	return dArgs, allKeys


def make_data_layers_proto(dPrms, nPrms, **kwargs):
	baseFile  = dPrms.paths.baseProto % nPrms.dataNetDefProto
	netDef    = mpu.ProtoDef(baseFile)
	#If net needs to be resumed
	if kwargs.has_key('resumeIter'):
		resumeIter = kwargs['resumeIter']
	else:
		resumeIter = 0
	#Modify the python layer parameters
	batchSz  = [nPrms.batchSize, 50]
	meanFile = sev2.get_mean_file(nPrms.meanFile) 
	for s, b in zip(['TRAIN', 'TEST'], batchSz):
		#The group files
		prmStr = ou.make_python_param_str({'batch_size': b,
					    'window_file': dPrms.paths.window[s.lower()],	 
							'im_root_folder': dPrms.paths.data.imFolder,
							'crop_size'  : nPrms.crpSz,
							'im_size'    : nPrms.ipImSz, 
              'jitter_amt' : nPrms.maxJitter,
							'resume_iter': resumeIter, 
							'mean_file': meanFile,
              'ncpu': nPrms.ncpu})
		netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % prmStr, phase=s)
	return netDef

def make_base_layers_proto(dPrms, nPrms, **kwargs):
	#Read the basefile and construct a net
	baseFile  = dPrms.paths.baseProto % nPrms.baseNetDefProto
	netDef    = mpu.ProtoDef(baseFile)
	return netDef 

def make_loss_layers_proto(dPrms, nPrms, lastTop, **kwargs):
	#Read the basefile and construct a net
	baseFile  = dPrms.paths.baseProto % nPrms.lossNetDefProto
	netDef    = mpu.ProtoDef(baseFile)
	if nPrms.lossNetDefProto in ['pascal_pose_loss_log_l1_layers']:
		lNames = ['az_reg_fc', 'el_reg_fc', 'az_cls_fc', 'el_cls_fc']
		#Set the name of the bottom
		for l in lNames:
			netDef.set_layer_property(l, 'bottom', '"%s"' % lastTop)
	else:
		raise Exception ('%s not found' % nPrms.lossNetDefProto)
	return netDef 

##
#Make the net def
def make_net_def(dPrms, nPrms, **kwargs):
	#Data protodef
	dataDef  = make_data_layers_proto(dPrms, nPrms, **kwargs)
	#Base net protodef
	baseDef  = make_base_layers_proto(dPrms, nPrms, **kwargs)
	#Get the name of the last top
	lastTop  = baseDef.get_last_top_name()
	print lastTop
	#Loss protodef
	lossDef  = make_loss_layers_proto(dPrms, nPrms, lastTop, **kwargs)
	#Merge al the protodefs
	return sev2._merge_defs([dataDef, baseDef, lossDef]) 


##
#Process the data and net parameters
def process_net_prms(**kwargs):
	'''
		net_prms_fn: The function to obtain net parameters
	'''
	nPrms, nKeys = net_prms(**kwargs)
	#Verify that no spurious keys have been added
	nKeysIp = [k for k in nPrms.keys() if not k in ['expStr']]
	assert set(nKeys)==set(nKeysIp), 'There are some spurious keys'
	return nPrms 

class ProcessPrms(object):
	def __init__(self, net_prms_fn):
		self.fn_ = net_prms_fn

	def process(self, **kwargs):
		nPrms, nKeys = self.fn_(**kwargs)
		#Verify that no spurious keys have been added
		nKeysIp = [k for k in nPrms.keys() if not k == 'expStr']
		assert set(nKeys)==set(nKeysIp), 'There are some spurious keys'
		return nPrms 


def setup_experiment_demo(debugMode=False, isRun=False):
	dPrms   = get_data_prms()
	nwFn    = process_net_prms
	ncpu = 0
	nwArgs  = {'ncpu': ncpu, 'lrAbove': None, 'preTrainNet':None}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': 10}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn=make_net_def, isLog=True)
	if isRun:
		exp.make()
		exp.run() 
	return exp 	 				


