##Records all the experiments I run
import street_params as sp
import street_exp as se

def smallnetv2_pool4_pose_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
																						deviceId=[0], isPythonLayer=False):
	prms  = sp.get_prms_pose(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
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
										deviceId=[0], isPythonLayer=False, isGray=False):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=45, rawImSz=256,
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


def smallnetv2_pool4_ptch_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
																		isPythonLayer=False, deviceId=[2]):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def smallnetv2_pool4_nrml_crp192_rawImSz256(isRun=False, isGray=False,
																			 numTrain=1e+7, deviceId=[0,1]):
	prms  = sp.get_prms_nrml(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=1.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def smallnetv2_pool4_nrml_crp192_rawImSz256_nojitter(isRun=False, isGray=False,
																			 numTrain=1e+7, deviceId=[0,1]):
	prms  = sp.get_prms_nrml(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, maxJitter=0)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000,
												 clip_gradients=10.0, debug_info=True)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def ptch_pose_exp2(isRun=False, deviceId=[1], numPoseStream=256, numPatchStream=256, numTrain=1e+7):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose', 'ptch'], 
											labelType=['quat', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100,
											numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
							 multiLossProto='v1', ptchStreamNum=numPatchStream,
							 poseStreamNum=numPoseStream)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=1.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


