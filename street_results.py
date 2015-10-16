import my_exp as me
import my_pycaffe_io as mpio
import my_pycaffe_utils as mpu
import street_exp as se
import my_pycaffe as mp
import caffe
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as scm
import vis_utils as vu
import rot_utils as ru
import pdb

def vis_results_pose(prms, cPrms, modelIter):
	#Initialize the network
	exp       = se.setup_experiment(prms, cPrms)
	modelFile = exp.get_snapshot_name(numIter=modelIter) 
	defFile   = exp.expFile_.def_
	net       = caffe.Net(defFile, modelFile, caffe.TEST)
	bSz       =	exp.get_layer_property('window_data', 'batch_size', phase='TEST')
	crpSz     = int(exp.get_layer_property('window_data', 'crop_size', phase='TEST'))
	muFile    = exp.get_layer_property('window_data', 'mean_file')[1:-1]
	mn        = mp.read_mean(muFile).transpose(1,2,0)
	st        = int((227 - 101)/2.0)
	print mn.shape 
	mn        = mn[st:st+101,st:st+101,:]
	mn        = mn[:,:,[2,1,0]]
	plt.ion()
	count      = 0
	numBatches = 5
	fig  = plt.figure()
	for nb in range(numBatches):
		#Generate results on test set		
		data = net.forward(['data', 'data_p', 'pose_fc', 'pose_label'])
		for b in range(int(bSz)):
			predLbl   = data['pose_fc'][b].squeeze()
			predEuler = ru.quat2euler(predLbl)  		
			gtLbl    = data['pose_label'][b].squeeze()
			gtEuler  = ru.quat2euler(gtLbl)
			tStr = 'GT- roll: %.2f, yaw: %.2f, pitch: %.2f\n'\
							+ 'PD- roll: %.2f, yaw: %.2f, pitch: %.2f'
			#pdb.set_trace()
			tStr = tStr % (gtEuler + predEuler)
			im1  = data['data'][b].transpose(1,2,0).squeeze()
			im2  = data['data_p'][b].transpose(1,2,0).squeeze()
			im1  = np.maximum(0,np.minimum(255,im1[:,:,[2,1,0]] + mn))
			im2  = np.maximum(0,np.minimum(255,im2[:,:,[2,1,0]] + mn))
			#pdb.set_trace()
			vu.plot_pairs(im1, im2, fig, figTitle=tStr)
			count += 1
			imName = exp.paths.testImVis % count
			plt.savefig(imName)
			
