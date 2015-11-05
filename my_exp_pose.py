##Records all the experiments I run
import street_params as sp
import street_exp as se

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

def smallnetv5_pose_crp192_fc5_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
								deviceId=[0], isPythonLayer=True, runNum=0, extraFc=None,
								numFc5=None, lrAbove=None):
	prms  = sp.get_prms_pose(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v5',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								numFc5=numFc5, extraFc=extraFc,
								lrAbove=lrAbove)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=1.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId, runNum=runNum)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


#Maximum 45 degree rotation
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


#Maximum 90 degree rotation
def smallnetv2_pool4_pose_euler_mx90_crp192_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=False, isGray=False, 
										extraFc=None):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=90, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	

#Maximum 90 degree rotation
def smallnetv5_fc5_pose_euler_mx90_crp192_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=True, isGray=False, 
										extraFc=None, lrAbove=None, numFc5=None, geoFence='dc-v2',
										numTest=1e+4):
	prms  = sp.get_prms(geoFence=geoFence, labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=90, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192, numTest=numTest)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v5',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, numFc5=numFc5, lrAbove=lrAbove)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	

#NO Maximum EULER ANGLE rotation
def smallnetv5_fc5_pose_euler_crp192_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=True, isGray=False, 
										extraFc=None, lrAbove=None, numFc5=None, numTest=1e+4):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=None, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192,
											numTest=numTest)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v5',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, numFc5=numFc5, lrAbove=lrAbove)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	

#NO Maximum EULER ANGLE rotation
def smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=True, isGray=False, 
										extraFc=None, lrAbove=None, numFc5=None):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l1'], maxEulerRot=None, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v5',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, numFc5=numFc5, lrAbove=lrAbove)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	



#Classifying Euler Angles
def smallnetv5_pool4_pose_classify_euler_crp192_rawImSz256(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=True, isGray=False,
										numFc5=512):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['classify'], nBins=[20], binTypes=['uniform'], 
											maxEulerRot=None, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v5',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
							  numFc5=numFc5)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0,
												debug_info=True)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


