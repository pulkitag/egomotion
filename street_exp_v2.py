import my_exp_config as mec
import os
from os import path as osp
import socket
from easydict import EasyDict as edict
import other_utils as ou
import my_pycaffe_utils as mpu

REAL_PATH = os.path.dirname(os.path.realpath(__file__))
HOST_NAME = socket.gethostname()
DEF_DB    = osp.join(REAL_PATH, 'exp-data/db-store/default-%s-%s-db.sqlite')
if 'ivb' in HOST_NAME:
	HOST_STR = 'nvCluster'
else:
	HOST_STR = HOST_NAME
DEF_DB    = DEF_DB % (HOST_STR, '%s')

##
#data directories depending on the host
def get_datadirs():
	if HOST_NAME == 'anakin':
		expDir  = '/data0/pulkitag/projBaxter/exp'
		dataDir = '/data0/pulkitag/data_sets/baxter_data/' 
	elif 'ivb' in HOST_NAME:
		expDir  = '/data/shared/pulkitag/projBaxter/exp'
		dataDir = '/data/shared/pulkitag/data_sets/baxter_data/' 
	else:
		raise Exception('Host not recognized')
	return expDir, dataDir

##
#get the mean file name
def get_mean_file(muPrefix):
	if muPrefix == '':
		return 'None'
	if muPrefix in ['imagenet']:
		_, dataDir = get_datadirs()
		muFile = osp.join(dataDir, muPrefix, 'mean_bgr.pkl')
	else:
		raise Exception('%s prefix for mean not recognized' % muPrefix)
	return muFile	

##
#Paths that are required for reading the data
def get_paths(dPrms=None):
	if dPrms is None:
		dPrms = data_prms()
	expDir, dataDir = get_datadirs()
	ou.mkdir(expDir)
	pth        = edict()
	#All the experiment paths
	pth.exp    = edict() 
	pth.exp.dr = expDir
	#Snapshots
	pth.exp.snapshot    = edict()
	pth.exp.snapshot.dr = osp.join(pth.exp.dr, 'snapshot')
	ou.mkdir(pth.exp.snapshot.dr)
	#Data files
	pth.data    = edict()
	pth.data.dr = osp.join(dataDir, dPrms.dataFolder)
	pth.baseProto = osp.join(REAL_PATH, 'base_files', '%s.prototxt')
	return pth	

##
#Parameters that govern what data is being used
def get_data_prms(dbFile=DEF_DB % 'data', **kwargs):
	dArgs   = mec.edict()
	dArgs.dataset = 'dc-v2'
	allKeys = dArgs.keys()  
	dArgs   = mpu.get_defaults(kwargs, dArgs)	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
	dArgs['paths']  = get_paths(dArgs) 
	return dArgs

##
#Parameters that govern the structure of the net
def net_prms(dbFile=DEF_DB % 'net', **kwargs):
	dArgs = mec.get_siamese_net_prms(dbFile, **kwargs)
	del dArgs['expStr']
	#Modify the baseNetDefProto
	dArgs.baseNetDefProto = 'smallnet_v5_window_siamese_fc5'
	if dArgs.batchSize is None:
		dArgs.batchSize = 128 
	#The amount of jitter in both the images
	dArgs.maxJitter = 0
	#The size of crop that should be cropped from the image
	dArgs.crpSz     = 192
	#the size to which the cropped image should be resized
	dArgs.ipImSz    = 101
	#The size of the fc layer if present
	dArgs.fcSz      = None
	##The mean file
	dArgs.meanFile  = ''
	dArgs.meanType  = None
	dArgs   = mpu.get_defaults(kwargs, dArgs, False)
	allKeys = dArgs.keys()	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
	return dArgs, allKeys


class ProcessPrms(object):
	def __init__(self, net_prms_fn):
		self.fn_ = net_prms_fn

	def process(self, **kwargs):
		nPrms, nKeys = self.fn_(**kwargs)
		#Verify that no spurious keys have been added
		nKeysIp = [k for k in nPrms.keys() if not k == 'expStr']
		assert set(nKeys)==set(nKeysIp), 'There are some spurious keys'
		return nPrms 

##
#Process the data and net parameters
def process_prms(**kwargs):
	'''
		net_prms_fn: The function to obtain net parameters
	'''
	nPrms, nKeys = net_prms(**kwargs)
	#Verify that no spurious keys have been added
	nKeysIp = [k for k in nPrms.keys() if not k == 'expStr']
	assert set(nKeys)==set(nKeysIp), 'There are some spurious keys'
	return nPrms 

##
#Make the net def
def make_net_def(dPrms, nPrms, **kwargs):
	#Read the basefile and construct a net
	baseFile  = dPrms.paths.baseProto % nPrms.baseNetDefProto
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
		prmStr = ou.make_python_param_str({'batch_size': b, 'before': 'image_before',
								'after': 'image_after', 'poke': 'pixel', 
								'root_folder': osp.join(dPrms.paths.data.dr, s.lower()),
								'crop_size'  : nPrms.crpSz, 'max_jitter': nPrms.maxJitter,
								'resume_iter': resumeIter, 
								'mean_file': meanFile, 'mean_type': nPrms.meanType,
								'poke_tfm_type': nPrms.pokeTfmType,
								'poke_nxGrid': nPrms.pokeNxGrid, 'poke_nyGrid': nPrms.pokeNyGrid,
								'poke_thGrid': nPrms.pokeThGrid})
		netDef.set_layer_property('data', ['python_param', 'param_str'], 
						'"%s"' % prmStr, phase=s)

	if nPrms.pokeTfmType == 'gridCls':
		netDef.set_layer_property('pred_loc', ['inner_product_param', 'num_output'],
				 '%d' % (nPrms.pokeNxGrid * nPrms.pokeNyGrid), phase='TRAIN')
		netDef.set_layer_property('pred_th', ['inner_product_param', 'num_output'],
				 '%d' % nPrms.pokeThGrid, phase='TRAIN')

	return netDef 
		  	

def setup_experiment_demo():
	dPrms   = get_data_prms()
	nwFn    = process_prms
	nwArgs  = {}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol'}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn=make_net_def) 
	return exp 	 				
