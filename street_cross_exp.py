##Records/Performs the experiment which evaluate the pose model on the patch task and
## vice-versa. 
import street_params as sp
import street_exp as se
import my_exp_ptch as mept
import my_exp_pose as mepo

def train_ptch_using_pose():
	ptPrms, ptCPrms = mept.smallnetv5_fc5_ptch_crp192_rawImSz256(numFc5=512, lrAbove='common_fc')
	poPrms, poCPrms = mepo.smallnetv5_pool4_pose_crp192_fc5_rawImSz256(numFc5=512)
	exp = se.setup_experiment_from_previous(poPrms, poCPrms, ptPrms, ptCPrms, srcModelIter=60000)
	return exp
	#Rename common_fc so that it is initialized randomly 

