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
import math
import my_pycaffe_io as mpio
import setup_nyu as snyu

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

##
#Parameters that govern what data is being used
def get_data_prms(dbFile=DEF_DB % 'nyu2_data', **kwargs):
	dArgs   = edict()
	dArgs.dataset = 'nyu2'
	allKeys = dArgs.keys()  
	dArgs   = mpu.get_defaults(kwargs, dArgs)	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
	dArgs['paths']  = snyu.get_paths()
	return dArgs

##
#Parameters that govern the structure of the net
def net_prms(dbFile=DEF_DB % 'nyu2_net', **kwargs):
	dArgs = mec.get_default_net_prms(dbFile, **kwargs)
	del dArgs['expStr']
	#The data NetDefProto
	dArgs.dataNetDefProto = 'data_layer_nyu2' 
	#the basic network architecture: baseNetDefProto
	dArgs.baseNetDefProto = 'doublefc-v1_window_fc6'
	#the loss layers:
	dArgs.lossNetDefProto = 'nyu2_loss_classify_layers'
	if dArgs.batchSize is None:
		dArgs.batchSize = 128 
	#The amount of jitter in both the images
	dArgs.maxJitter = 0
	#The size of crop that should be cropped from the image
	dArgs.cropScale  = 0.9
	#the size to which the cropped image should be resized
	dArgs.ipImSz    = 101
	##The mean file
	dArgs.meanFile  = ''
	dArgs.meanType  = None
	dArgs.opLrMult  = None
	dArgs   = mpu.get_defaults(kwargs, dArgs, False)
	allKeys = dArgs.keys()	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
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
						  'split'      : s.lower(),
							'crop_scale' : nPrms.cropScale,
							'im_size'    : nPrms.ipImSz, 
              'jitter' : nPrms.maxJitter,
							'mean_file': meanFile})
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
	lNames = ['sfn_fc']
	for l in lNames:
		netDef.set_layer_property(l, 'bottom', '"%s"' % lastTop)
		if nPrms.opLrMult is not None:
			netDef.set_layer_property(l, ['param', 'lr_mult'], '%f' % nPrms.opLrMult)
			netDef.set_layer_property(l, ['param_$dup$', 'lr_mult'], '%f' % (2 * nPrms.opLrMult))
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
	#print lastTop
	#Loss protodef
	lossDef  = make_loss_layers_proto(dPrms, nPrms, lastTop, **kwargs)
	#Merge al the protodefs
	netDef = sev2._merge_defs([dataDef, baseDef, lossDef]) 
	if nPrms.lrAbove is not None:
		netDef.set_no_learning_until(nPrms.lrAbove)
	return netDef


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
	ncpu    = 0
	nwArgs  = {'lrAbove': None, 'preTrainNet':None}
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
