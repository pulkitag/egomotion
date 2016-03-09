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
import street_exp_v2 a sev2

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

##
#Parameters that govern what data is being used
def get_data_prms(dbFile=DEF_DB % 'data', **kwargs):
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
def net_prms(dbFile=DEF_DB % 'net', **kwargs):
	dArgs = mec.get_net_prms(dbFile, **kwargs)
	del dArgs['expStr']
	#The data NetDefProto
	dArgs.dataNetDefProto = 'data_layer_pascal' 
	#the basic network architecture: baseNetDefProto
	dArgs.baseNetDefProto = 'smallnet-v5_window_siamese_fc5'
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
	meanFile = get_mean_file(nPrms.meanFile) 
	for s, b in zip(['TRAIN', 'TEST'], batchSz):
		#The group files
		prmStr = ou.make_python_param_str({'batch_size': b, 
							'im_root_folder': imFolder,
							'crop_size'  : nPrms.crpSz,
							'im_size'    : nPrms.ipImSz, 
              'jitter_amt' : nPrms.maxJitter,
							'resume_iter': resumeIter, 
							'mean_file': meanFile,
              'ncpu': nPrms.ncpu,
		netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % prmStr, phase=s)
		#Rename the top corresponding to the labels
		lbName = '"%s_label"' % lbInfo.lb['type']
		top2 = mpu.make_key('top', ['top'])
		netDef.set_layer_property('window_data', top2, lbName, phase=s)
	#Split the pair data according to the labels
	baseFile  = dPrms.paths.baseProto % '%s_layers'
	baseFile  = baseFile % lbInfo.lb['type']
	splitDef  = mpu.ProtoDef(baseFile)
	return sev2._merge_defs([netDef, splitDef])

def make_base_layers_proto(dPrms, nPrms, **kwargs):
	#Read the basefile and construct a net
	baseFile  = dPrms.paths.baseProto % nPrms.baseNetDefProto
	netDef    = mpu.ProtoDef(baseFile)
	return netDef 

def make_loss_layers_proto(dPrms, nPrms, **kwargs):
	#Read the basefile and construct a net
	baseFile  = dPrms.paths.baseProto % nPrms.lossNetDefProto
	netDef    = mpu.ProtoDef(baseFile)
	return netDef 

##
#Make the net def
def make_net_def(dPrms, nPrms, **kwargs):
	#Data protodef
	dataDef  = make_data_layers_proto(dPrms, nPrms, **kwargs)
	#Base net protodef
	baseDef  = make_base_layers_proto(dPrms, nPrms, **kwargs)
	#Loss protodef
	lossDef  = make_loss_layers_proto(dPrms, nPrms, **kwargs)
	#Merge al the protodefs
	return sev2._merge_defs([dataDef, baseDef, lossDef]) 


def setup_experiment_demo(debugMode=False, isRun=False):
	dPrms   = get_data_prms()
	nwFn    = sev2.process_net_prms
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


