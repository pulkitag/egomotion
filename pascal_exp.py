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
	#Get the label-stats
	pth.exp.labelStats = osp.join(pth.exp.dr, 'label_stats.pkl')
	#info label for the experiment
	pth.exp.lbInfo     = osp.join(pth.exp.dr, 'label_info', dPrms.expStr, 'lbinfo.pkl') 
	ou.mkdir(osp.dirname(pth.exp.lbInfo))
	#Results
	pth.exp.results = edict()
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
	#Window files stores theta in degrees
	pth.window.train = osp.join(windowDr, 'euler_train_pascal3d_imSz%d_pdSz%d.txt')
	pth.window.test  = osp.join(windowDr, 'euler_test_pascal3d_imSz%d_pdSz%d.txt')
	pth.window.train = pth.window.train % (dPrms.imCutSz, dPrms.imPadSz)
	pth.window.test  = pth.window.test %  (dPrms.imCutSz, dPrms.imPadSz)
	return pth	

##
#Find the bin index
def find_bin_index(bins, val):
	'''
		bins[0]  is the minimum possible value
		bins[-1] is the maximum possible value
	'''
	assert (val > bins[0] - 1e-6), val
	assert (val < bins[-1] + 1e-6), val
	idx = np.where(val < bins)[0]
	if len(idx)==0:
		return len(bins)-2
	else:
		return max(0, idx[0]-1)

#format radians
def format_radians(theta):
	theta = np.mod(theta, 2*np.pi)
	if theta > np.pi:
		theta = -(2 * np.pi - theta)
	return theta
	
#Format window file label 
def format_label(theta, lbInfo, mu=None, sd=None, bins=None):
	'''
		Input
		  theta: is in radian
    Output:
      flag: 1 - go in anticlockwise
			      0 - go in clockwise
			theta: the absolute value of theta
	'''
	#For classification
	if lbInfo['anglePreProc'] == 'classify':
		theta = format_radians(theta)
		assert bins is not None
		return find_bin_index(bins, theta), None
	#For regression
	assert lbInfo['angleFormat'] == 'radian'
	theta = np.mod(theta, 2*np.pi)
	theta = math.radians(theta)
	if theta > np.pi:
		theta = -(2 * np.pi - theta)
		flag  = 0
	else:
		flag  = 1
	#Perform preprocessing
	if lbInfo['anglePreProc'] in ['mean_sub', 'amean_sub', 'med_sub', 'amed_sub']:
		assert mu is not None
		theta = theta - mu
	elif lbInfo['anglePreProc'] in ['zscore']:
		assert mu is not None
		assert sd is not None
		theta = (theta - mu)/sd
	elif lbInfo['anglePreProc'] is None:
		pass
	else:
		raise Exception('pre-proc %s not understood' % lbInfo['anglePreProc']) 
	return flag, theta

#Unformat the formatted label
def unformat_label(theta, flag, lbInfo, mu=None, sd=None, bins=None):
	assert lbInfo['angleFormat'] == 'radian'
	#Classification label
	if lbInfo['anglePreProc'] == 'classify':
		assert bins is not None
		mn, mx = bins[theta], bins[theta+1]
		return (mn + mx)/2.0
	#Regression labels
	if lbInfo['anglePreProc'] in ['mean_sub', 'amean_sub', 'med_sub', 'amed_sub']:
		assert mu is not None
		theta = theta + mu
	elif lbInfo['anglePreProc'] in ['zscore']:
		assert mu is not None
		assert sd is not None
		theta = theta * sd + mu
	elif lbInfo['anglePreProc'] is None:
		pass
	else:
		raise Exception('pre-proc %s not understood' % lbInfo['anglePreProc']) 
	if flag == 0:
		theta = -theta
	return theta	

##
#Parameters that govern what data is being used
def get_data_prms(dbFile=DEF_DB % 'pascal_data', **kwargs):
	dArgs   = edict()
	dArgs.dataset = 'pascal'
	dArgs.imCutSz = 256
	dArgs.imPadSz = 36
	dArgs.angleFormat  = 'radian'
	dArgs.anglePreProc = 'mean_sub'
	dArgs.nAzBins = None
	dArgs.nElBins = None
	allKeys = dArgs.keys()  
	dArgs   = mpu.get_defaults(kwargs, dArgs)	
	if dArgs.anglePreProc == 'classify':
		assert dArgs.nAzBins is not None
		assert dArgs.nElBins is not None
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
	dArgs['paths']  = get_paths(dArgs)
	dArgs.azBins = None
	dArgs.elBins = None
	if dArgs.nAzBins is not None:
		dArgs.azBins = np.linspace(-np.pi, np.pi, dArgs.nAzBins+1)
	if dArgs.nElBins is not None:
		dArgs.elBins = np.linspace(-np.pi, np.pi, dArgs.nElBins+1)
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
	dArgs.crpSz     = 240
	#the size to which the cropped image should be resized
	dArgs.ipImSz    = 101
	##The mean file
	dArgs.meanFile  = ''
	dArgs.meanType  = None
	dArgs.ncpu      = 3
	dArgs.opLrMult  = None
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
							'lb_info_file': dPrms.paths.exp.lbInfo,
							'crop_size'  : nPrms.crpSz,
							'im_size'    : nPrms.ipImSz, 
              'jitter_amt' : nPrms.maxJitter,
							'resume_iter': resumeIter, 
							'mean_file': meanFile,
              'ncpu': nPrms.ncpu})
		netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % prmStr, phase=s)
	lbKeys = ['angleFormat', 'anglePreProc', 'azBins', 'elBins']
	lb     = edict()
	for lk in lbKeys:
		lb[lk] = dPrms[lk]
	lbInfo = pickle.load(open(dPrms.paths.exp.labelStats, 'r'))
	for lk in lbInfo.keys():
		lb[lk] = lbInfo[lk]
	pickle.dump(lb, open(dPrms.paths.exp.lbInfo, 'w'))
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
	elif nPrms.lossNetDefProto in ['pascal_pose_loss_classify_layers']:
			netDef.set_layer_property('azimuth_fc', ['inner_product_param', 'num_output'],
        dPrms.nAzBins)
			netDef.set_layer_property('azimuth_fc', 'bottom', '"%s"' % lastTop)
			netDef.set_layer_property('elevation_fc', ['inner_product_param', 'num_output'],
        dPrms.nElBins)
			netDef.set_layer_property('elevation_fc', 'bottom', '"%s"' % lastTop)
			lNames = ['azimuth_fc', 'elevation_fc']
	else:
		raise Exception ('%s not found' % nPrms.lossNetDefProto)
	if nPrms.opLrMult is not None:
		for l in lNames:
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

def compute_label_stats(dPrms=None):
	if dPrms is None:	
		dPrms = get_data_prms()
	if osp.exists(dPrms.paths.exp.labelStats):
		print ('Label Stats already computed')
		return
	randState = np.random.RandomState(5)
	wFile     = dPrms.paths.window.train		
	wFid      = mpio.GenericWindowReader(wFile)
	lbls      = wFid.get_all_labels()
	N         = lbls.shape[0]
	perm      = randState.permutation(N)
	perm      = perm[0:int(0.2*N)]
	lbls      = lbls[perm,:]
	print (lbls.shape)
	mu,  sd   = np.mean(lbls, 0), np.std(lbls,0)
	aMu, aSd  = np.mean(np.abs(lbls),0), np.std(np.abs(lbls),0) 
	md, aMd   = np.median(lbls,0), np.median(np.abs(lbls),0)	
	print (mu, sd, aMu, aSd)
	print (md, aMd)
	stats = edict()
	stats['mu'], stats['sd'] = mu, sd
	stats['aMu'], stats['aSd'] = aMu, aSd
	stats['md'], stats['aMd']  = md, aMd
	wFid.close()
	pickle.dump(stats, open(dPrms.paths.exp.labelStats, 'w'))
