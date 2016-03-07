import street_exp_v2 as sev2
import street_process_data as spd
import street_config as cfg
import pickle
import numpy as np
import street_label_utils as slu
import my_exp_pose_grps as mepg

REAL_PATH = cfg.REAL_PATH

#First lets make a proper test set :) 
def make_test_set(dPrms, numTest=100000):
	listName = dPrms['paths'].exp.other.grpList % 'test'
	data     = pickle.load(open(listName, 'r'))	
	grpDat   = []
	grpCount = []
	numGrp   = 0
	for i,g in enumerate(data['grpFiles']):
		grpDat.append(pickle.load(open(g, 'r'))['groups'])
		grpCount.append(len(grpDat[i]))
		print ('Groups in %s: %d' % (g, grpCount[i]))
		numGrp += grpCount[i]
	print ('Total number of groups: %d' % numGrp)
	grpSampleProb = [float(i)/float(numGrp) for i in grpCount]
	randSeed  = 7
	randState = np.random.RandomState(randSeed) 
	elms      = []
	for t in range(numTest):
		if np.mod(t,5000)==1:
			print(t)	
		breakFlag = False
		while not breakFlag:
			rand   =  randState.multinomial(1, grpSampleProb)
			grpIdx =  np.where(rand==1)[0][0]
			ng     =  randState.randint(low=0, high=grpCount[grpIdx])
			grp    =  grpDat[grpIdx][ng]
			l1     =  randState.permutation(grp.num)[0]
			l2     =  randState.permutation(grp.num)[0]
			if l1==l2:
				rd = randState.rand()
				#Sample the same image rarely
				if rd < 0.85:
					continue
			elm = [grp.folderId, grp.crpImNames[l1], grp.crpImNames[l2]]
			lb  = slu.get_pose_delta(dPrms['lbPrms'].lb, grp.data[l1].rots,
            grp.data[l2].rots, grp.data[l1].pts.camera,
            grp.data[l2].pts.camera)
			lb  = np.array(lb)
			elm.append(lb)
			elms.append(elm)
			breakFlag = True
	pickle.dump({'testData': elms}, open(dPrms.paths.exp.other.testData, 'w'))
	return elms


def make_deploy_net(exp, numIter=60000):
 	exp.make_deploy(dataLayerNames=['window_data'], imSz=exp.netPrms_['ipImSz'])
	modelName = exp.get_snapshot_name(numIter=numIter)
	if not osp.exists(modelName):
		print ('ModelFile doesnot exist')
		return None
	net       = mp.MyNet(exp.files_['netDefDeploy'], modelName)
	#Set preprocessing
	net.set_preprocess(ipName='window_data', chSwap=None, noTransform=True)
	return net
	
def demo_test():
	exp = mepg.simple_euler_dof2_dcv2_doublefcv1(gradClip=30,
        stepsize=60000, base_lr=0.001, gamma=0.1)	
	net = make_deploy_net(exp)
	if net is None:
		return
