##Records all the experiments I run
import street_params as sp
import street_exp as se

def smallnetv2_pool4_ptch_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
					isPythonLayer=False, deviceId=[2], batchsize=256,
					resumeIter=0, extraFc=None, lrAbove=None):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, lrAbove=lrAbove)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId,
								resumeIter=resumeIter)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


def smallnetv2_fc5_ptch_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
					isPythonLayer=True, deviceId=[2], batchsize=256,
					resumeIter=0, extraFc=None, numFc5=512, runNum=0):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, numFc5=numFc5)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=10000, 
											clip_gradients=10.0, debug_info=True)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId,
								resumeIter=resumeIter, runNum=runNum)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


def smallnetv5_fc5_ptch_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
					isPythonLayer=True, deviceId=[2], batchsize=256,
					resumeIter=0, extraFc=None, numFc5=512, runNum=0,
					lrAbove=None):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v5',
							 concatLayer='fc5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, numFc5=numFc5, lrAbove=lrAbove)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=10000, 
											clip_gradients=10.0, debug_info=True)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId,
								resumeIter=resumeIter, runNum=runNum)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	

def smallnetv6_pool4_ptch_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
					isPythonLayer=True, deviceId=[2], batchsize=256,
					resumeIter=0, extraFc=None, numConv4=64, runNum=0):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v6',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc, numConv4=numConv4)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=10000, 
											clip_gradients=10.0, debug_info=True)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId,
								resumeIter=resumeIter, runNum=runNum)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	


def matchnet_ptch_crp192_imSz64_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
					isPythonLayer=True, deviceId=[2], batchsize=256,
					resumeIter=0, extraFc=None):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=64, netName='matchnet',
							 concatLayer='pool5', lossWeight=10.0,
								randCrop=False, concatDrop=False,
								isGray=isGray, isPythonLayer=isPythonLayer,
								extraFc=extraFc)
	lPrms = se.get_lr_prms(batchsize=batchsize, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId,
								resumeIter=resumeIter)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	return prms, cPrms	

