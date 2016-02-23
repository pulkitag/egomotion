##Records all the pose experiments that I am running
#There was a bug in pose computation done in earlier experiments
#this file only contains the new experiments
#
import street_params as sp
import street_exp as se

#NO Maximum EULER ANGLE rotation
def smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=True, isGray=False, 
										extraFc=None, lrAbove=None, numFc5=None):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['l1'], maxEulerRot=None, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192, isV2=True,
											labelNrmlz='zscore')
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


#NO Maximum EULER ANGLE rotation with LogL1
def smallnetv5_fc5_pose_euler_crp192_rawImSz256_loss_logl1(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=True, isGray=False, 
										extraFc=None, lrAbove=None, numFc5=None, 
										stepsize=10000):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler'],
											lossType=['logl1'], maxEulerRot=None, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192, isV2=True,
											labelNrmlz='zscore')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v5',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, numFc5=numFc5, lrAbove=lrAbove)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=stepsize, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


#No Maximum Euler Angle and 5-DOF
def smallnetv5_fc5_pose_euler_5dof_crp192_rawImSz256_lossl1(isRun=False, numTrain=1e+7, 
										deviceId=[0], isPythonLayer=True, isGray=False, 
										extraFc=None, lrAbove=None, numFc5=None):
	prms  = sp.get_prms(geoFence='dc-v2', labels=['pose'], labelType=['euler-5dof'],
											lossType=['l1'], maxEulerRot=None, rawImSz=256,
											splitDist=100, numTrain=numTrain, crpSz=192, isV2=True,
											labelNrmlz='zscore')
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

