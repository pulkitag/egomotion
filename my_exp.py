##Records all the experiments I run
import street_params as sp
import street_exp as se

def run_smallnet_pool4_pose():
	prms  = sp.get_prms_pose(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1])
	exp   = se.make_experiment(prms, cPrms)
	exp.run()	

def run_smallnet_pool4_nrml():
	prms  = sp.get_prms_nrml(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1,2,3])
	exp   = se.make_experiment(prms, cPrms)
	exp.run()	 
