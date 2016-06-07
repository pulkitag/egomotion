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

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

##
#get the mean file name
def get_mean_file(muPrefix):
	if muPrefix == '' or muPrefix is None:
		return 'None'

	if muPrefix == 'imagenet':
		_, dataDir = get_datadirs()
		muFile = osp.join(dataDir, muPrefix, 'mean_bgr.pkl')
	elif muPrefix == 'imagenet_proto':
		muFile    = osp.join(cfg.pths.data0, 'caffe_models/ilsvrc2012_mean.binaryproto')
	else:
		raise Exception('%s prefix for mean not recognized' % muPrefix)
	return muFile	


def get_folder_paths(folderId, splitPrms=None, isAlign=False, hostName=None):
	if isAlign:
		alignStr = 'aligned'
	else:
		alignStr = 'unaligned'
	if splitPrms is None:
		splitPrms = get_trainval_split_prms()
	#is hostName is not None means we are getting paths
  #of a different machine - so dont create directories
	if hostName is not None:
		mkDir = False
	else:
		mkDir = True	
	cPaths,_   = cfg.get_paths(hostName) 
	paths    = edict()
	paths.dr   = cPaths.folderProc % folderId
	paths.procTarDr = cPaths.folderProcTar % folderId
	paths.prefix     = osp.join(paths.dr, 'prefix.pkl')
	paths.prefixAlign = osp.join(paths.dr, 'prefixAlign.pkl')
	paths.prePerGrp   = osp.join(paths.dr, 'prePerGrp.pkl')
	#List of targetgroups in ordered format - necessary as ordering matters
	#ordering can be used to split the data into train/val/test as points
	#closer in the ordering are physically close to each other
	paths.targetGrpList = osp.join(paths.dr, 'targetGrpList.pkl')
	paths.targetGrps = osp.join(paths.dr, 'targetGrps.pkl')
	#List of aligned stuff
	paths.targetGrpListAlign = osp.join(paths.dr, 'targetGrpListAlign.pkl')
	paths.targetGrpsAlign    = osp.join(paths.dr, 'targetGrpsAlign.pkl')
	#path for storing the cropped images
	paths.crpImStr   = 'imCrop/imSz%s' % '%d'
	paths.crpImStrAlign   = 'imCrop/imSz%s-align' % '%d'
	paths.crpImPath  = osp.join(paths.dr, paths.crpImStr)
	paths.crpImPathAlign  = osp.join(paths.dr, paths.crpImStrAlign)
	paths.crpImPathTar    = osp.join(paths.procTarDr, paths.crpImStr) + '.tar'
	paths.crpImPathAlignTar  = osp.join(paths.procTarDr, paths.crpImStrAlign) + '.tar'
	#Split the sets
	paths.grpSplits  = edict()
	#The derived directory for storing derived info
	paths.deriv = edict()
	#3 %s correspond to - splitPrms.pStr, aligned/nonaligned, folderId
	paths.deriv.grps  = cPaths.folderDerivDir %\
            ('grpSplitStore', osp.join(splitPrms.pStr, alignStr,
             folderId))
	paths.deriv.grpsTar = cPaths.folderDerivDirTar % \
            ('grpSplitStore', osp.join(splitPrms.pStr,  alignStr, folderId + '.tar'))
	dirName = osp.basename(paths.deriv.grpsTar)
	if mkDir:
		ou.mkdir(dirName)
	for s in ['train', 'val', 'test']:
		paths.grpSplits[s]  = osp.join(paths.deriv.grps, 
						'groups_%s.pkl' % s)
		dirName = osp.basename(paths.grpSplits[s])
		if mkDir:
			ou.mkdir(dirName)
	paths.trainvalSplitGrpKeys = osp.join(paths.deriv.grps, 
									 'splits-keys.pkl')
	if mkDir:
		ou.mkdir(paths.dr)
		ou.mkdir(paths.procTarDr)
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
	if dPrms['isAlign']:
		pth.exp.other.grpList = osp.join(pth.exp.other.dr,
					'group_list_aligned_%s_%s.pkl' % (dPrms['splitPrms']['pStr'], '%s')) 
	else:
		pth.exp.other.grpList = osp.join(pth.exp.other.dr,
					'group_list_%s_%s.pkl' % (dPrms['splitPrms']['pStr'], '%s')) 
	ou.mkdir(pth.exp.other.dr)
	pth.exp.other.lbInfo   = osp.join(pth.exp.other.dr, 'label_info_%s.pkl')
	#Save the test set
	pth.exp.other.testData = osp.join(pth.exp.other.dr, 'test_data_%s.pkl')
	pth.exp.other.testData = pth.exp.other.testData % dPrms['lbPrms'].get_lbstr()
	#Pose stats
	pth.exp.other.poseStats = osp.join(pth.exp.other.dr, 'pose_stats_%s.pkl') 
	pth.exp.other.poseStats = pth.exp.other.poseStats % dPrms['lbPrms'].get_lbstr() 
	#Save the predictions
	pth.exp.results = edict()
	pth.exp.results.dr   = osp.join(pth.exp.dr, 'results', '%s')
	pth.exp.results.file = osp.join(pth.exp.results.dr, 'iter%d.pkl') 
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
	dArgs.isAlign = True
	allKeys = dArgs.keys()  
	dArgs   = mpu.get_defaults(kwargs, dArgs)	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs)
	dArgs['splitPrms'] = tvPrms
	dArgs['lbPrms']    = lbPrms
	dArgs['paths']  = get_paths(dArgs) 
	return dArgs

##
#Parameters that govern the structure of the net
def net_prms(dbFile=DEF_DB % 'net', **kwargs):
	dArgs = mec.get_siamese_net_prms(dbFile, **kwargs)
	del dArgs['expStr']
	#The data NetDefProto
	dArgs.dataNetDefProto = 'data_layer_groups' 
	#the basic network architecture: baseNetDefProto
	dArgs.baseNetDefProto = 'smallnet-v5_window_siamese_fc5'
	#the loss layers:
	dArgs.lossNetDefProto = 'pose_loss_log_l1_layers'
	if dArgs.batchSize is None:
		dArgs.batchSize = 128 
	#The amount of jitter in both the images
	dArgs.maxJitter = 0
	#The amount of roll jitter to apply to the images
	dArgs.maxRollJitter = None
	#The size of crop that should be cropped from the image
	dArgs.crpSz     = 192
	#the size to which the cropped image should be resized
	dArgs.ipImSz    = 101
	#The size of the fc layer if present
	dArgs.fcSz      = None
	dArgs.fcName    = 'fc5'
	##The mean file
	dArgs.meanFile  = ''
	dArgs.meanType  = None
	dArgs.ncpu      = 3
	dArgs.readSingleGrp = False
	dArgs   = mpu.get_defaults(kwargs, dArgs, False)
	allKeys = dArgs.keys()	
	dArgs['expStr'] = mec.get_sql_id(dbFile, dArgs, ignoreKeys=['ncpu', 'readSingleGrp'])
	return dArgs, allKeys


##
#Merge the definition of multiple layers
def _merge_defs(defs): 
	allDef = copy.deepcopy(defs[0])
	for d in defs[1:]:
		setNames = ['TRAIN', 'TEST']
		for s in setNames:
			trNames = d.get_all_layernames(phase=s)
			for t in trNames:
				trLayer = d.get_layer(t, phase=s)		
				allDef.add_layer(t, trLayer, phase=s)
	return allDef


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
		#Make the label info file
		lbInfo = dPrms['lbPrms']
		lbDict = copy.deepcopy(lbInfo.lb)
		lbDict['lbSz']      = lbInfo.get_lbsz()
		lbDict['statsFile'] = dPrms.paths.exp.other.poseStats 
		lbFile = dPrms.paths.exp.other.lbInfo % lbInfo.get_lbstr()
		pickle.dump({'lbInfo': lbDict}, open(lbFile, 'w'))
		#The group files
		if s == 'TEST':
			grpListFile = dPrms.paths.exp.other.grpList % 'val'
		else:
			grpListFile = dPrms.paths.exp.other.grpList % 'train'
		#The python parameters
		if dPrms['isAlign']:
			imFolder = osp.join(cfg.pths.folderProc, 'imCrop', 'imSz256-align')
		else:
			imFolder = osp.join(cfg.pths.folderProc, 'imCrop', 'imSz256')
		prmStr = ou.make_python_param_str({'batch_size': b, 
							'im_root_folder': imFolder,
							'grplist_file': grpListFile,
						  'lbinfo_file':  lbFile, 
							'crop_size'  : nPrms.crpSz,
							'im_size'    : nPrms.ipImSz, 
              'jitter_amt' : nPrms.maxJitter,
							'random_roll_max': nPrms.maxRollJitter,
							'resume_iter': resumeIter, 
							'mean_file': meanFile,
              'ncpu': nPrms.ncpu,
							'is_single_grp': nPrms.readSingleGrp})
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
	return _merge_defs([netDef, splitDef])


def make_base_layers_proto(dPrms, nPrms, **kwargs):
	#Read the basefile and construct a net
	baseFile  = dPrms.paths.baseProto % nPrms.baseNetDefProto
	netDef    = mpu.ProtoDef(baseFile)
	if nPrms.fcSz is not None:
		netDef.set_layer_property(nPrms.fcName,
       ['inner_product_param', 'num_output'],
				'%d' % (nPrms.fcSz), phase='TRAIN')
	return netDef 


def make_loss_layers_proto(dPrms, nPrms, **kwargs):
	#Read the basefile and construct a net
	baseFile  = dPrms.paths.baseProto % nPrms.lossNetDefProto
	netDef    = mpu.ProtoDef(baseFile)
	fcLayerName = '%s_fc' % dPrms.lbPrms.lb['type']
	lbSz        = dPrms.lbPrms.get_lbsz()
	netDef.set_layer_property(fcLayerName,
            ['inner_product_param', 'num_output'], '%d' % lbSz, phase='TRAIN')
	
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
	return _merge_defs([dataDef, baseDef, lossDef]) 
			  	

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
	posePrms = slu.PosePrms()
	dPrms   = get_data_prms(lbPrms=posePrms)
	nwFn    = process_net_prms
	if debugMode:
		ncpu = 0
	else:
		ncpu = 2
	nwArgs  = {'ncpu': ncpu}
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


def make_group_list_file(dPrms=None):
	if dPrms is None:
		dPrms = get_data_prms()
	fName  = osp.join(REAL_PATH, 'geofence', '%s_list.txt')
	fName  = fName % dPrms['dataset']
	fid    = open(fName, 'r')
	fList  = [l.strip() for l in fid.readlines()]
	fid.close()
	fStore = spd.FolderStore()
	setNames = ['train', 'val', 'test']
	for s in setNames: 	
		grpListFileName = dPrms['paths'].exp.other.grpList
		grpListFileName = grpListFileName %  s
		print ('Saving to %s' % grpListFileName)
		grpFiles    = []
		for f in fList: 		
			assert fStore.is_present(f)
			folderId   = fStore.get_id(f)
			folderPath = get_folder_paths(folderId,
            dPrms['splitPrms'], isAlign=dPrms['isAlign']) 
			grpFiles.append(folderPath.grpSplits[s])
		pickle.dump({'grpFiles': grpFiles}, open(grpListFileName, 'w'))


def save_pose_stats(dPrms=None, nPrms=None):
	if dPrms is None:
		dPrms = get_data_prms()
	if nPrms is None:
		nPrms = get_net_prms()
	listName = dPrms['paths'].exp.other.grpList % 'val'
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
	lbs      = []
	for t in range(50000):
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
			if nPrms.maxRollJitter is None:
				grp1Rot, grp2Rot = grp.data[l1].rots, grp.data[l2].rots
			else:
				grp1Rot = copy.deepcopy(grp.data[l1].rots)
				grp2Rot =  copy.deepcopy(grp.data[l2].rots)
				rollJitter = slu.get_roll_jitter(nPrms.maxRollJitter)
				#if t < 3:
				#	import IPython; IPython.embed()
				grp1Rot[2] = grp1Rot[2] + rollJitter[0]
				grp2Rot[2] = grp2Rot[2] + rollJitter[1]
				grp1Rot    = tuple(grp1Rot)
				grp2Rot    = tuple(grp2Rot)
			lb  = slu.get_pose_delta(dPrms['lbPrms'].lb, grp1Rot,
            grp2Rot, grp.data[l1].pts.camera,
            grp.data[l2].pts.camera)
			lb  = np.array(lb)
			lbs.append(lb.reshape((1,)+lb.shape))
			breakFlag = True
	lbs = np.concatenate(lbs)
	mu  = np.mean(lbs,0)
	sd  = np.std(lbs,0)
	md  = np.median(lbs,0)
	mx  = np.max(lbs, 0)
	mn  = np.min(lbs, 0)
	aMx = np.max(np.abs(lbs),0)
	aMn = np.min(np.abs(lbs),0)	
	pickle.dump({'mu':mu, 'sd':sd, 'md':md, 'mx':mx,'mn':mn,
   'aMx':aMx, 'aMn':aMn}, open(dPrms.paths.exp.other.poseStats, 'w'))
	return lbs


#Check if all group files are present or not
def verify_group_list_files(dPrms):
	setNames = ['train', 'val', 'test']
	for s in setNames: 	
		grpListFileName = dPrms['paths'].exp.other.grpList % s
		data = pickle.load(open(grpListFileName, 'r'))
		for fName in data['grpFiles']:
			if not osp.exists(fName):
				print ('%s doesnot exists' % fName)
			

	
