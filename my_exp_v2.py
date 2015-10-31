##Records all the experiments I run
import street_params as sp
import street_exp as se
import my_exp_ptch as mept

def smallnetv2_pool4_pose_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
																						deviceId=[0], isPythonLayer=False, runNum=0):
	prms  = sp.get_prms_pose(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=1.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId, runNum=runNum)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def smallnetv2_pool4_pose_euler_mx45_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=False, isGray=False):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=45, rawImSz=256,
											splitDist=100, numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


def smallnetv2_pool4_pose_euler_mx45_crp192_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=False, isGray=False, extraFc=None,
										resumeIter=0):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=45, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer, 
								extraFc=extraFc)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId, resumeIter=resumeIter)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	

def smallnetv2_pool4_pose_euler_mx90_crp192_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=False, isGray=False):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=90, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


#POSE CLASSIFICATION
def smallnetv2_pool4_pose_classify_euler_mx45_crp192_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=False, isGray=False):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['classify'], nBins=[10], binTypes=['uniform'], 
											maxEulerRot=45, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	



def smallnetv2_pool4_nrml_crp192_rawImSz256(isRun=False, isGray=False,
																			 numTrain=1e+7, deviceId=[0],
																			 makeNrmlUni=0.002, isPythonLayer=True):
	prms  = sp.get_prms_nrml(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain, nrmlMakeUni=makeNrmlUni)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=1.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def smallnetv2_pool4_nrml_crp192_rawImSz256_nojitter(isRun=False, isGray=False,
																			 numTrain=1e+7, deviceId=[0],
																			 makeNrmlUni=0.002, isPythonLayer=True):
	prms  = sp.get_prms_nrml(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain, nrmlMakeUni=makeNrmlUni)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, maxJitter=0, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000,
												 clip_gradients=10.0, debug_info=True)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def smallnetv2_pool4_nrml_classify_crp192_rawImSz256_nojitter(isRun=False, isGray=False,
																			 numTrain=1e+7, deviceId=[0,1],
																			 makeNrmlUni=0.002, isPythonLayer=True):
	prms  = sp.get_prms(labels=['nrml'], labelType=['xyz'],
						lossType=['classify'], nBins=[20], binTypes=['uniform'], 
						geoFence='dc-v2', crpSz=192,
						rawImSz=256, splitDist=100,
						numTrain=numTrain, nrmlMakeUni=makeNrmlUni)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, maxJitter=0, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000,
												 clip_gradients=10.0, debug_info=True)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def ptch_pose_euler_mx45_exp1(isRun=False, deviceId=[1], numTrain=1e+7, batchsize=256,
								 extraFc=None, isPythonLayer=True):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['euler', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain, maxEulerRot=45,
											nBins=[None, None], binTypes=[None, None])
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
							 multiLossProto=None, extraFc=extraFc,
							 isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=20000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def ptch_pose_euler_mx90_exp1(isRun=False, deviceId=[1], numTrain=1e+7, batchsize=256,
								 extraFc=None, isPythonLayer=True):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['euler', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain, maxEulerRot=90,
											nBins=[None, None], binTypes=[None, None])
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
							 multiLossProto=None, extraFc=extraFc,
							 isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=20000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def ptch_pose_euler_mx90_alexnet_exp1(isRun=False, deviceId=[1], numTrain=1e+7, batchsize=256,
								 extraFc=None, isPythonLayer=True):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['euler', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain, maxEulerRot=90,
											nBins=[None, None], binTypes=[None, None])
	nPrms = se.get_nw_prms(imSz=101, netName='alexnet',
							 concatLayer='conv5', lossWeight=10.0,
							 multiLossProto=None, extraFc=extraFc,
							 isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=20000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def ptch_pose_euler_mx90_smallnet_v2_fc5_exp1(isRun=False, deviceId=[1],
						 numTrain=1e+7, batchsize=256, extraFc=None, isPythonLayer=True,
					   numFc5=None, numCommonFc=None):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['euler', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain, maxEulerRot=90,
											nBins=[None, None], binTypes=[None, None])
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
					 concatLayer='fc5', lossWeight=10.0,
							 multiLossProto=None, extraFc=extraFc,
							 isPythonLayer=isPythonLayer, numFc5=numFc5,
							 numCommonFc=numCommonFc)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=20000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def ptch_pose_euler_mx90_streetnet_fc5_exp1(isRun=False, deviceId=[1],
						 numTrain=1e+7, batchsize=256, extraFc=None, isPythonLayer=True,
					   numFc5=None, numCommonFc=None):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['euler', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain, maxEulerRot=90,
											nBins=[None, None], binTypes=[None, None])
	nPrms = se.get_nw_prms(imSz=101, netName='streetnet',
					 concatLayer='fc5', lossWeight=10.0,
							 multiLossProto=None, extraFc=extraFc,
							 isPythonLayer=isPythonLayer, numFc5=numFc5,
							 numCommonFc=numCommonFc)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=20000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	



def ptch_pose_euler_mx45_exp1_from_ptch(isRun=False, deviceId=[1], 
								 numTrain=1e+7, batchsize=256,
								 extraFc=None, isPythonLayer=True,
								 poseModelIter=10000):

	#srcPrms, srcCPrms = smallnetv2_pool4_pose_euler_mx45_crp192_rawImSz256(isRun=False,
	#				isPythonLayer=True, extraFc=512)

	srcPrms, srcCPrms = mept.smallnetv2_pool4_ptch_crp192_rawImSz256(isRun=False,
					isPythonLayer=True)

	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['euler', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain, maxEulerRot=45,
											nBins=[None, None], binTypes=[None, None])
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
							 multiLossProto=None, extraFc=extraFc,
							 isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=20000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)

	exp = se.make_experiment_from_previous(srcPrms, srcCPrms, prms, cPrms,
						srcModelIter=poseModelIter)
	if isRun:
		exp.run()
	return prms, cPrms



def ptch_pose_euler_mx45_exp2(isRun=False, deviceId=[1], numPoseStream=256,
								 numPatchStream=256, numTrain=1e+7, batchsize=256,
								 extraFc=None, isPythonLayer=True):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['euler', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain, maxEulerRot=45,
											nBins=[None, None], binTypes=[None, None])
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
							 multiLossProto='v1', ptchStreamNum=numPatchStream,
							 poseStreamNum=numPoseStream, extraFc=extraFc,
							 isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=20000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


