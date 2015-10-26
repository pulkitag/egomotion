##Records all the experiments I run
import street_params as sp
import street_exp as se

def smallnetv2_pool4_pose_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
																						deviceId=[0]):
	prms  = sp.get_prms_pose(geoFence='dc-v2', crpSz=192,
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

def smallnetv2_pool4_ptch_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

