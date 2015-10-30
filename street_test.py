import read_liberty_patches as rlp
import my_exp_ptch as mept
import street_exp as se
import my_pycaffe as mp
import my_pycaffe_utils as mpu
import numpy as np
import caffe

def modify_params(paramStr, key, val):
	params = paramStr.strip().split('--')
	newStr = ''
	for i,p in enumerate(params):
		if len(p) ==0:
			continue
		if not(len(p.split()) == 2):
			continue
		k, v = p.split()
		if k.strip() == key:
			v = val
		newStr = newStr + '--%s %s ' % (k,v)
	return newStr

def get_fpr(recall, pdScore, gtLabel):
	N = sum(gtLabel==1)
	M = sum(gtLabel==0)
	assert(N+M == gtLabel.shape[0])
	idx = np.argsort(pdScore)
	#Sort in Decreasing Order
	pdScore = pdScore[idx[::-1]]
	gtLabel = gtLabel[idx[::-1]]
	posCount = np.cumsum(gtLabel==1)/float(N)
	threshIdx = np.where((posCount > recall)==True)[0][0]
	pdLabel   = pdScore >= 0.5
	pdLabel   = pdLabel[0:threshIdx]
	numPos    = np.sum(pdLabel==1)
	fpr       = (threshIdx - float(numPos))/float(threshIdx)
	return fpr
	
		
def test_ptch(prms, cPrms, modelIter):
	exp       = se.setup_experiment(prms, cPrms)
	modelFile = exp.get_snapshot_name(modelIter)
	libPrms   = rlp.get_prms()
	wFile     = libPrms.paths.wFile

	netDef    = mpu.ProtoDef(exp.files_['netdef'])
	paramStr  = netDef.get_layer_property('window_data', 'param_str')[1:-1]
	paramStr  = modify_params(paramStr, 'source', wFile)
	paramStr  = modify_params(paramStr, 'root_folder', libPrms.paths.jpgDir)
	netDef.set_layer_property('window_data', ['python_param', 'param_str'], 
						'"%s"' % paramStr)
	defFile = 'test-files/ptch_liberty_test.prototxt'
	netDef.write(defFile)
	net = caffe.Net(defFile, modelFile, caffe.TEST)

	gtLabel, pdScore = [], []
	for i in range(10):
		data = net.forward(['ptch_label','ptch_fc'])
		gtLabel.append(data['ptch_label'].squeeze())
		score   = np.exp(data['ptch_fc'])
		score   = score/(np.sum(score,1).reshape(score.shape[0],1))
		pdScore .append(score[:,1])
	gtLabel = np.concatenate(gtLabel)
	pdScore = np.concatenate(pdScore)
	return gtLabel, pdScore

	
