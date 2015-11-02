##Records/Performs the experiment which evaluate the pose model on the patch task and
## vice-versa. 
import street_params as sp
import street_exp as se
import my_exp_ptch as mept
import my_exp_pose as mepo

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

def train_pose_using_ptch():
	poPrms, poCPrms = mepo.smallnetv5_fc5_pose_euler_mx90_crp192_rawImSz256(numFc5=512, 
																							lrAbove='common_fc')
	ptPrms, ptCPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(numFc5=512)
	exp = se.setup_experiment_from_previous(poPrms, poCPrms, ptPrms, ptCPrms, srcModelIter=60000)
	return exp
	#Rename common_fc so that it is initialized randomly 

