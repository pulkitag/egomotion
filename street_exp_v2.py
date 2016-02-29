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

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

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


def get_folder_paths(folderId, splitPrms):
	cPaths   = cfg.pths 
	paths    = edict()
	paths.dr = cPaths.folderProc % folderId
	ou.mkdir(paths.dr)
	paths.prefix     = osp.join(paths.dr, 'prefix.pkl')
	paths.prePerGrp  = osp.join(paths.dr, 'prePerGrp.pkl')
	#List of targetgroups in ordered format - necessary as ordering matters
	#ordering can be used to split the data into train/val/test as points
	#closer in the ordering are physically close to each other
	paths.targetGrpList = osp.join(paths.dr, 'targetGrpList.pkl')
	paths.targetGrps = osp.join(paths.dr, 'targetGrps.pkl')
	#path for storing the cropped images
	paths.crpImStr   = 'imCrop/imSz%s' % '%d'
	paths.crpImPath  = osp.join(paths.dr, paths.crpImStr)
	#Split the sets
	paths.trainvalSplit = osp.join(paths.dr, 
									 'splits-%s.pkl' % splitPrms.pStr)
	paths.grpSplits  = edict()
	for s in ['train', 'val', 'test']:
		paths.grpSplits[s]  = osp.join(paths.dr, 
										 'groups_%s_%s.pkl' % (s, splitPrms.pStr))
	return paths


##
#Paths that are required for reading the data
def get_paths(dPrms=None):
	if dPrms is None:
		dPrms = data_prms()
	expDir, dataDir = cfg.pths.expDir, cfg.pths.mainDataDr
	ou.mkdir(expDir)
	pth        = edict()
	#All the experiment paths
	pth.exp    = edict() 
	pth.exp.dr = expDir
	#Snapshots
	pth.exp.snapshot    = edict()
	pth.exp.snapshot.dr = osp.join(pth.exp.dr, 'snapshot')
	ou.mkdir(pth.exp.snapshot.dr)
	#group lists
	pth.exp.other         = edict()
	pth.exp.other.dr      = osp.join(pth.exp.dr, 'others')
	pth.exp.other.grpList = osp.join(pth.exp.other.dr, 'group_list_%s_%s.pkl') 
	ou.mkdir(pth.exp.other.dr)
	
	#Data files
	pth.data    = edict()
	pth.data.dr  = dataDir
	pth.baseProto = osp.join(REAL_PATH, 'base_files', '%s.prototxt')
	return pth	

#Forming the trainval splits
def get_trainval_split_prms(**kwargs):
	dArgs = edict()
	dArgs.trnPct = 85
	dArgs.valPct = 5
	dArgs.tePct  = 10
	assert (dArgs.trnPct + dArgs.valPct + dArgs.tePct == 100)
	#The minimum distance between groups belonging to two sets
	#in meters (m)
	dArgs.minDist = 100
	dArgs = ou.get_defaults(kwargs, dArgs, True)
	dArgs.pStr = 'trn%d_val%d_te%d_dist%d' % (dArgs.trnPct, dArgs.valPct,
								dArgs.tePct, dArgs.minDist)
	return dArgs	

##
#Parameters that govern what data is being used
def get_data_prms(dbFile=DEF_DB % 'data', lbPrms=None, tvPrms=None, **kwargs):
	if lbPrms is None:
		lbPrms = slu.PosePrms()
	if tvPrms is None:
		tvPrms = get_trainval_split_prms()
	dArgs   = mec.edict()
	dArgs.dataset = 'dc-v2'
	dArgs.lbStr   = lbPrms.get_lbstr()
	dArgs.tvStr   = tvPrms.pStr
	allKeys = dArgs.keys()  
	dArgs   = mpu.get_defaults(kwargs, dArgs)	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
	dArgs['paths']  = get_paths(dArgs) 
	dArgs['splitPrms'] = tvPrms
	dArgs['lbPrms']    = lbPrms
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


##
#Process the data and net parameters
def process_net_prms(**kwargs):
	'''
		net_prms_fn: The function to obtain net parameters
	'''
	nPrms, nKeys = net_prms(**kwargs)
	#Verify that no spurious keys have been added
	nKeysIp = [k for k in nPrms.keys() if not k == 'expStr']
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
	posePrms = slu.PosePrms()
	dPrms   = get_data_prms(lbPrms=posePrms)
	nwFn    = process_net_prms
	nwArgs  = {}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol'}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn=make_net_def) 
	return exp 	 				


def make_group_list_file(dPrms):
	fName  = osp.join(REAL_PATH, 'geofence', '%s_list.txt')
	fName  = fName % dPrms['dataset']
	fid    = open(fName, 'r')
	fList  = [l.strip() for l in fid.readlines()]
	fid.close()
	fStore = spd.FolderStore()
	setNames = ['train', 'val', 'test']
	for s in setNames: 	
		grpListFileName = dPrms['paths'].exp.other.grpList
		grpListFileName = grpListFileName % (dPrms['splitPrms']['pStr'], s)
		print ('Saving to %s' % grpListFileName)
		grpFiles    = []
		for f in fList: 		
			assert fStore.is_present(f)
			folderId   = fStore.get_id(f)
			folderPath = get_folder_paths(folderId, dPrms['splitPrms']) 
			grpFiles.append(folderPath.grpSplits[s])
		pickle.dump({'grpFiles': grpFiles}, open(grpListFileName, 'w'))
			 
