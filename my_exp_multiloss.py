##Record the multiloss experiments that I am running
import street_params as sp
import street_exp as se

'''
#### Experiment Descriptions ######
ptch_pose_exp1
Net: smallnetv2
		 ptch_fc and pose_fc from common_fc

ptch_pose_exp2
Net: smallnetv2
		 ptch_stream and pose_stream from common_fc
		 ptch_fc and pose_fc from their respective Streams

###
'''

def ptch_pose_exp1(isRun=False, deviceId=[1]):
	prms  = sp.get_prms(geoFence='dc-v1', labels=['pose', 'ptch'], 
											labelType=['quat', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def ptch_pose_exp2(isRun=False, deviceId=[1], numPoseStream=256, numPatchStream=256):
	prms  = sp.get_prms(geoFence='dc-v1', labels=['pose', 'ptch'], 
											labelType=['quat', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192, splitDist=100)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
							 multiLossProto='v1', ptchStreamNum=numPatchStream,
							 poseStreamNum=numPoseStream)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=deviceId)
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


