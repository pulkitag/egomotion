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


########### NETWORK V2 ######################################
def smallnetv2_pool4_ptch(isRun=False):
	prms  = sp.get_prms_ptch(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def smallnetv2_pool4_ptch_pose_crp192_rawImSz256(isRun=False):
	prms  = sp.get_prms(geoFence='dc-v1', labels=['pose', 'ptch'], 
											labelType=['quat', 'wngtv'],
											lossType=['l2', 'classify'], labelFrac=[0.5,0.5],
											rawImSz=256, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

# %%%%%%%%%%%%%%%%%%%%%%%%%%% POSE %%%%%%%%%%%%%%%%%%%%%%%%%%% #
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

def smallnetv2_pool4_pose_euler_mx45(isRun=False):
	prms  = sp.get_prms(geoFence='dc-v1', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=45)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=1.0)
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def smallnetv2_1_pool4_pose_euler_mx45_crp192(isRun=False):
	prms  = sp.get_prms(geoFence='dc-v1', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=45, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
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


def smallnetv2_pool4_pose_crp192(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1', crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def smallnetv2_pool4_pose_crp192_rawImSz(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1', crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	



#Same as above but with random cropping
def smallnetv2_pool4_pose_crp192_randcrp(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1', crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=True)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

#Same as above but with dropouts
def smallnetv2_pool4_pose_crp192_randcrp_drop(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1', crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0,
								randCrop=True, concatDrop=True)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


#Pose and patch networks
def smallnetv2_pool4_pose_ptch_crp192(isRun=False):
	prms  = sp.get_prms(geoFence='dc-v1', crpSz=192,
						labels=['pose', 'ptch'],
						labelType=['quat', 'wngtv'],
						lossType=['l2', 'classify'],
						ptchPosFrac=0.5)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	



########### NETWORK V3 ######################################
def smallnetv3_pool4_pose(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1')
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v3',
							 concatLayer='pool4')
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0,1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	

def smallnetv3_pool4_pose_euler_mx45_crp192(isRun=False):
	prms  = sp.get_prms(geoFence='dc-v1', labels=['pose'], labelType=['euler'],
											lossType=['l2'], maxEulerRot=45, crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v3',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256, clip_gradients=1.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[1])
	if isRun:
		exp   = se.make_experiment(prms, cPrms)
		exp.run()
	else:
		return prms, cPrms	


def smallnetv3_pool4_pose_crp192(isRun=False):
	prms  = sp.get_prms_pose(geoFence='dc-v1', crpSz=192)
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v3',
							 concatLayer='pool4', lossWeight=10.0)
	lPrms = se.get_lr_prms(batchsize=256, stepsize=10000, clip_gradients=10.0)
	cPrms = se.get_caffe_prms(nPrms, lPrms, deviceId=[0])
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
