##Records all the experiments I run
import street_params as sp
import street_exp as se

def smallnet_pool4_pose(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def smallnetv2_pool4_pose(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

##Run the smallnet for pose but with random cropping
def smallnetv2_pool4_pose_randcrp(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', randCrop=True)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=5000)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1,2,3])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	



def run_smallnet_pool4_nrml(isRun=False):
	prms  = sp.get_prms_nrml(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1,2,3])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()	
	else:
		return prms, cPrms 

def run_smallnetv2_pool4_nrml(isRun=False):
	prms  = sp.get_prms_nrml(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1,2,3])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()	
	else:
		return prms, cPrms 
