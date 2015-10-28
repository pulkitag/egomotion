##Records all the experiments I run
import street_params as sp
import street_exp as se

def smallnetv2_pool4_ptch_crp192_rawImSz256(isRun=False, isGray=False, numTrain=1e+7,
					isPythonLayer=False, deviceId=[2], batchsize=256,
					resumeIter=0, extraFc=None):
	prms  = sp.get_prms_ptch(geoFence='dc-v2', crpSz=192,
													 rawImSz=256, splitDist=100,
													 numTrain=numTrain)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
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

