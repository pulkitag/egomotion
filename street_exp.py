import os.path as osp
import numpy as np
import street_utils as su
import my_pycaffe_utils as mpu
from easydict import EasyDict as edict
import copy

##
# Parameters required to specify the n/w architecture
def get_nw_prms(**kwargs):
	dArgs = edict()
	dArgs.netName     = 'alexnet'
	dArgs.concatLayer = 'fc6'
	dArgs.concatDrop  = False
	dArgs.contextPad  = 24
	dArgs.imSz        = 227
	dArgs.imgntMean   = True
	dArgs = mpu.get_defaults(kwargs, dArgs)
	expStr = 'net-%s_cnct-%s_cnctDrp%d_contPad%d_imSz%d_imgntMean%d'\
						%(dArgs.netName, dArgs.concatLayer, dArgs.concatDrop, 
							dArgs.contextPad,
							dArgs.imSz, dArgs.imgntMean)
	dArgs.expStr = expStr 
	return dArgs 

##
# Parameters that specify the learning
def get_lr_prms(**kwargs):	
	dArgs = edict()
	dArgs.batchsize = 128
	dArgs.stepsize  = 20000	
	dArgs.base_lr   = 0.001
	dArgs.max_iter  = 250000
	dArgs.gamma     = 0.5
	dArgs.weight_decay = 0.0005 
	dArgs  = mpu.get_defaults(kwargs, dArgs)
	#Make the solver 
	solArgs = edict({'test_iter': 100, 'test_interval': 1000,
						 'snapshot': 1000, 'debug_info': 'true'})
	for k in dArgs.keys():
		if k in ['batchsize']:
			continue
		solArgs[k] = copy.deepcopy(dArgs[k])
	dArgs.solver = mpu.make_solver(**solArgs)	
	expStr = 'batchSz%d_stepSz%.0e_blr%.5f_mxItr%.1e_gamma%.2f_wdecay%.6f'\
					 % (dArgs.batchsize, dArgs.stepsize, dArgs.base_lr,
							dArgs.max_iter, dArgs.gamma, dArgs.weight_decay)
	dArgs.expStr = expStr
	return dArgs 

##
# Parameters for fine-tuning
def get_finetune_prms(**kwargs):
	'''
		sourceModelIter: The number of model iterations of the source model to consider
		fine_max_iter  : The maximum iterations to which the target model should be trained.
		lrAbove        : If learning is to be performed some layer. 
		fine_base_lr   : The base learning rate for finetuning. 
 		fineRunNum     : The run num for the finetuning.
		fineNumData    : The amount of data to be used for the finetuning. 
		fineMaxLayer   : The maximum layer of the source n/w that should be considered.  
	''' 
	dArgs = edict()
	dArgs.base_lr = 0.001
	dArgs.runNum  = 1
	dArgs.maxLayer = None
	dArgs.lrAbove  = None
	dArgs.dataset  = 'sun'
	dArgs.maxIter  = 40000
	dArgs.extraFc     = False
	dArgs.extraFcDrop = False
	dArgs.sourceModelIter = 60000 
	dArgs = mpu.get_defaults(kwargs, dArgs)
 	return dArgs 


def get_caffe_prms(nwPrms, lrPrms, finePrms=None, isScratch=True, deviceId=1): 
	caffePrms = edict()
	caffePrms.deviceId  = deviceId
	caffePrms.isScratch = isScratch
	caffePrms.nwPrms    = copy.deepcopy(nwPrms)
	caffePrms.lrPrms    = copy.deepcopy(lrPrms)
	caffePrms.finePrms  = copy.deepcopy(finePrms)

	expStr = nwPrms.expStr + '/' + lrPrms.expStr
	if finePrms is not None:
		expStr = expStr + '/' + finePrms.expStr
	caffePrms['expStr'] = expStr
	caffePrms['solver'] = lrPrms.solver
	return caffePrms


def get_default_caffe_prms(deviceId=1):
	nwPrms = get_nw_prms()
	lrPrms = get_lr_prms()
	cPrms  = get_caffe_prms(nwPrms, lrPrms, deviceId=deviceId)
	return cPrms


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

##
# The proto definitions for the loss
def make_loss_proto(prms, cPrms):
	baseFilePath = prms.paths.baseNetsDr
	if prms.isSiamese and 'nrml' in prms.labelNames:
		defFile = osp.join(baseFilePath, 'nrml_loss_layers.prototxt')
		nrmlDef1 = mpu.ProtoDef(defFile)
		nrmlDef2 = mpu.ProtoDef(defFile)
		#Structure the two defs
		nrmlDef1.set_layer_property('nrml_fc', 'name', '"nrml_1_fc"')
		nrmlDef1.set_layer_property('nrml_1_fc','top', '"nrml_1_fc"')
		nrmlDef2.set_layer_property('nrml_fc', 'name', '"nrml_2_fc"')
		nrmlDef2.set_layer_property('nrml_2_fc','top', '"nrml_2_fc"')
		#Merge the two defs			 	
		lbDef = _merge_defs(nrmlDef1, nrmlDef2)
	elif 'nrml' in prms.labelNames:
		defFile = osp.join(baseFilePath, 'nrml_loss_layers.prototxt')
		lbDef   = mpu.ProtoDef(defFile)
	return lbDef	

##
#Adapt the ProtoDef for the data layers
#Helper function for setup_experiment
def _adapt_data_proto(protoDef, prms, cPrms):
	if prms.isAligned:
		rootDir = '/data0/pulkitag/data_sets/streetview/raw/ssd105/Amir/WashingtonAligned/'
	else:
		raise Exception('rootDir is not defined')
	#Get the source file for the train and test layers
	protoDef.set_layer_property('window_data', ['generic_window_data_param', 'source'],
			'"%s"' % prms['paths']['windowFile']['train'], phase='TRAIN')
	protoDef.set_layer_property('window_data', ['generic_window_data_param', 'source'],
			'"%s"' % prms['paths']['windowFile']['test'], phase='TEST')

	#Set the root folder
	protoDef.set_layer_property('window_data', ['generic_window_data_param', 'root_folder'],
			'"%s"' % rootDir, phase='TRAIN')
	protoDef.set_layer_property('window_data', ['generic_window_data_param', 'root_folder'],
			'"%s"' % rootDir, phase='TEST')
	
	#Set the batch size
	protoDef.set_layer_property('window_data', ['generic_window_data_param', 'batch_size'],
			'%d' % cPrms.lrPrms.batchsize , phase='TRAIN')

	#if prms['randomCrop']:
	protoDef.set_layer_property('window_data', ['generic_window_data_param', 'random_crop'],
		'false', phase='TRAIN')
	protoDef.set_layer_property('window_data', ['generic_window_data_param', 'random_crop'],
		'false', phase='TEST')

	#Set the mean file
	if cPrms.nwPrms.imgntMean:
		if prms.isSiamese:
			fName = '/data0/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto'
		else:
			fName = '/data0/pulkitag/caffe_models/ilsvrc2012_mean_for_siamese.binaryproto'
		for p in ['TRAIN', 'TEST']:
			protoDef.set_layer_property('window_data', ['transform_param', 'mean_file'],
				'"%s"' % fName, phase=p)
	

##
#The proto definitions for the data
def make_data_proto(prms, cPrms):
	baseFilePath = prms.paths.baseNetsDr
	dataFile     = osp.join(baseFilePath, 'data_layers.prototxt')
	dataDef      = mpu.ProtoDef(dataFile)
	if len(prms.labelNames)==1:
			#Modify the label and data top names
		if prms.labelNames[0]=='nrml':
			for ph in ['TRAIN', 'TEST']:
				dataDef.set_layer_property('window_data', 'top', '"data"', phase=ph)
				top2 = mpu.make_key('top', ['top'])
				dataDef.set_layer_property('window_data', top2, '"nrml_label"', phase=ph)
		elif prms.labelNames[0]=='pose':
			pass
		elif prms.labelNames[0]=='ptch':
			pass
	else:
		#Add slicing of labels	
		sliceFile = '%s_layers.protoxt' % prms.labelNameStr
		sliceDef  = mpu.ProtoDef(osp.join(baseFilePath, sliceFile))
		dataDef   = _merge_defs(dataDef, sliceDef)	
	#Set to the new window files
	_adapt_data_proto(dataDef, prms, cPrms)
	return dataDef

##
#Setup the experiment
def setup_experiment(prms, cPrms):
	baseFilePath = prms.paths.baseNetsDr
	#Get the protodef for the n/w architecture
	if prms.isSiamese:
		netFileStr = '%s_window_siamese_%s.prototxt'
	else:
		netFileStr = '%s_window_%s.prototxt'
	netFile = netFileStr % (cPrms.nwPrms.netName,
												 cPrms.nwPrms.concatLayer) 
	netFile = osp.join(baseFilePath, netFile)
	netDef  = mpu.ProtoDef(netFile)
	#Data protodef
	dataDef  = make_data_proto(prms, cPrms)
	#Loss protodef
	lossDef  = make_loss_proto(prms, cPrms)
	#Merge all defs
	protoDef = _merge_defs([dataDef, netDef, lossDef])
	#Get the solver definition file
	solDef   = cPrms['solver']
	#Experiment Object	
	caffeExp = get_experiment_object(prms, cPrms)
	caffeExp.init_from_external(solDef, protoDef)
	return caffeExp

def make_experiment(prms, cPrms, isFine=False, resumeIter=None, 
										srcModelFile=None, srcDefFile=None):
	'''
		Specifying the srcModelFile is a hack to overwrite a model file to 
		use with pretraining. 
	'''
	if isFine:
		caffeExp = setup_experiment_finetune(prms, cPrms, srcDefFile=srcDefFile)
		if srcModelFile is None:
			#Get the model name from the source experiment.
			srcCaffeExp  = setup_experiment(prms, cPrms)
			if cPrms['fine']['modelIter'] is not None:
				modelFile = srcCaffeExp.get_snapshot_name(cPrms['fine']['modelIter'])
			else:
				modelFile = None
	else:
		caffeExp  = setup_experiment(prms, cPrms)
		modelFile = None

	if resumeIter is not None:
		modelFile = None

	if srcModelFile is not None:
		modelFile = srcModelFile

	caffeExp.make(modelFile=modelFile, resumeIter=resumeIter)
	return caffeExp	



def get_experiment_object(prms, cPrms):
	caffeExp = mpu.CaffeExperiment(prms['expName'], cPrms['expStr'], 
							prms['paths']['expDir'], prms['paths']['snapDir'],
						  deviceId=cPrms['deviceId'])
	return caffeExp



