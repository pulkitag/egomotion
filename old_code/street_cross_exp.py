##Records/Performs the experiment which evaluate the pose model on the patch task and
## vice-versa. 
import street_params as sp
import street_exp as se
import my_exp_ptch as mept
import my_exp_pose as mepo
import my_exp_v2 as mev2

def train_ptch_using_pose(isRun=False, deviceId=[0]):
	ptPrms, ptCPrms = mept.smallnetv2_pool4_ptch_crp192_rawImSz256(isPythonLayer=True, 
																				lrAbove='common_fc', deviceId=deviceId)
	poPrms, poCPrms = mepo.smallnetv2_pool4_pose_euler_mx90_crp192_rawImSz256(isPythonLayer=True,																													extraFc=512)
	exp, modelFile = se.setup_experiment_from_previous(poPrms, poCPrms, 
										ptPrms, ptCPrms, srcModelIter=60000)
	#Rename common_fc so that it is initialized randomly
	exp.expFile_.netDef_.rename_layer('common_fc', 'common_fc_new') 
	if isRun:
		exp.make(modelFile=modelFile)
		exp.run()
	return exp

def train_pose_using_ptch(isRun=False, deviceId=[0]):
	poPrms, poCPrms = mepo.smallnetv5_fc5_pose_euler_mx90_crp192_rawImSz256(numFc5=512, 
												lrAbove='common_fc', isPythonLayer=True, deviceId=deviceId)
	ptPrms, ptCPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(numFc5=512,
																 isPythonLayer=True)
	exp, modelFile = se.setup_experiment_from_previous(ptPrms, ptCPrms,
																 poPrms, poCPrms, srcModelIter=60000)
	#Rename common_fc so that it is initialized randomly
	exp.expFile_.netDef_.rename_layer('common_fc', 'common_fc_new') 
	if isRun:
		exp.make(modelFile=modelFile)
		exp.run()
	return exp

def train_ptch_using_pose_fc5(isRun=False, deviceId=[0]):
	poPrms, poCPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256(numFc5=512, 
											isPythonLayer=True)
	ptPrms, ptCPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(numFc5=512,
													isPythonLayer=True, lrAbove='common_fc', deviceId=deviceId)
	exp, modelFile = se.setup_experiment_from_previous(poPrms, poCPrms,
																 ptPrms, ptCPrms, srcModelIter=60000)
	#Rename common_fc so that it is initialized randomly
	exp.expFile_.netDef_.rename_layer('common_fc', 'common_fc_new') 
	if isRun:
		exp.make(modelFile=modelFile)
		exp.run()
	return exp


def train_ptch_using_ptch_lt5(isRun=False, deviceId=[0]):
	#The target experiment is to peform ptch matching on general angles
	tgtPrms, tgtCPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(isPythonLayer=True, 
																				lrAbove='common_fc', deviceId=deviceId)
	#The source experiment is ptch matching with positives only from euler angles lt 5
	srcPrms, srcCPrms = mept.smallnetv5_fc5_ptch_euler_mx5_crp192_rawImSz256(numFc5=512)
	exp, modelFile = se.setup_experiment_from_previous(srcPrms, srcCPrms, 
										tgtPrms, tgtCPrms, srcModelIter=36000)
	#Rename common_fc so that it is initialized randomly
	exp.expFile_.netDef_.rename_layer('common_fc', 'common_fc_new') 
	if isRun:
		exp.make(modelFile=modelFile)
		exp.run()
	return exp

def train_ptch_using_ptch_lt5_pose_all(isRun=False, deviceId=[0]):
	#The target experiment is to peform ptch matching on general angles
	tgtPrms, tgtCPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(isPythonLayer=True, 
																				lrAbove='common_fc', deviceId=deviceId)
	#The source experiment is ptch matching with positives only from euler angles lt 5
	srcPrms, srcCPrms = mev2.ptch_euler_mx5_pose_euler_smallnet_v5_fc5_exp1(numFc5=512)
	exp, modelFile = se.setup_experiment_from_previous(srcPrms, srcCPrms, 
										tgtPrms, tgtCPrms, srcModelIter=36000)
	#Rename common_fc so that it is initialized randomly
	exp.expFile_.netDef_.rename_layer('common_fc', 'common_fc_new') 
	if isRun:
		exp.make(modelFile=modelFile)
		exp.run()
	return exp

