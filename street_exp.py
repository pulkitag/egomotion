import os.path as osp
import numpy as np
import street_utils as su
import my_pycaffe_utils as mpu

##
# Parameters required to specify the n/w architecture
def get_nw_prms(**kwargs):
	dArgs = edict()
	dArgs.concatLayer = 'fc6'
	dArgs.concatDrop  = False
	dArgs.contextPad  = 24
	dArgs.imSz        = 227
	dArgs.imgntMean   = False
	dArgs = mpu.get_defaults(kwargs, dArgs)
	expStr = 'cnct-%s_cnctDrp%d_contPad%d_imSz%d_imgntMean%d'\
						% (dArgs.concatLayer, dArgs.concatDrop, dArgs.contextPad,
							 dArgs.imSz, dArgs.imgntMean)
	dArgs.expStr = expStr 
	return dArgs 

##
# Parameters that specify the learning
def get_lr_prms(**kwargs):	
	dArgs = edict()
	dArgs.batchSz = 128
	dArgs.stepSz  = 20000	
	dArgs  = mpu.get_defaults(kwargs, dArgs)
	expStr = 'batchSz%d_stepSz%d' % (dArgs.batchSz, dArgs.stepSz)
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

	expStr = []
	expStr.append('con-%s' % concatLayer)
	if isScratch:
		expStr.append('scratch')
	if concatDrop:
		expStr.append('con-drop')
	expStr.append('pad%d' % contextPad)
	expStr.append('imS%d' % imSz)	

	if convConcat:
		expStr.append('con-conv')
	
	if isMySimple:
		expStr.append('mysimple')

	if contrastiveMargin is not None:
		expStr.append('ct-margin%d' % contrastiveMargin)

	if isFineTune:
		if fineDataSet=='sun':
			assert (fineMaxIter is None) and (stepsize is None)
			#These will be done automatically.
			if imSz==227 or imSz==256:
				sunImSz = 256
				muFile = '"%s"' % '/data1/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto'
			else:
				sunImSz = 128
				muFile = '"%s"' % '/data1/pulkitag/caffe_models/ilsvrc2012_mean_imSz128.binaryproto'
			caffePrms['fine']['muFile'] = muFile
			print "Using mean from: ", muFile 
			sunPrms     = ps.get_prms(numTrainPerClass=fineNumData, runNum=fineRunNum, imSz=sunImSz)
			numCl       = 397
			numTrainEx  = numCl * fineNumData  
			maxEpoch    = 30
			numSteps    = 2
			epochIter   = np.ceil(float(numTrainEx)/batchSz)
			caffePrms['stepsize'] = int(maxEpoch * epochIter)
			caffePrms['fine']['max_iter'] = int(numSteps * caffePrms['stepsize']) 
			caffePrms['fine']['gamma'] = 0.1
			caffePrms['fine']['prms']  = sunPrms
			expStr.append('small-data')

		expStr.append(fineDataSet)
		if sourceModelIter is not None:
			expStr.append('mItr%dK' % int(sourceModelIter/1000))
		else:
			expStr.append('scratch')	
		if lrAbove is not None:
				expStr.append('lrAbv-%s' % lrAbove)
		expStr.append('bLr%.0e' % fine_base_lr)
		expStr.append('run%d' % fineRunNum)
		expStr.append('datN%.0e' % fineNumData)
		if fineMaxLayer is not None:
			expStr.append('mxl-%s' % fineMaxLayer)
		if addDrop:
			expStr.append('drop')
		if extraFc:
			expStr.append('exFC')
		if imgntMean:
			expStr.append('muImgnt')

	expStr = ''.join(s + '_' for s in expStr)
	expStr = expStr[0:-1]
	caffePrms['expStr'] = expStr
	caffePrms['solver'] = get_solver(caffePrms, isFine=isFineTune)
	return caffePrms


def get_experiment_object(prms, cPrms):
	caffeExp = mpu.CaffeExperiment(prms['expName'], cPrms['expStr'], 
							prms['paths']['expDir'], prms['paths']['snapDir'],
						  deviceId=cPrms['deviceId'])
	return caffeExp



