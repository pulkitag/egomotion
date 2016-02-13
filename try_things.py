import my_exp_pose as mepo
import street_exp as se
import my_pycaffe_utils as mpu
import vis_utils as vu
import numpy as np
import my_exp_v2 as mev2
import my_pycaffe_io as mpio
import numpy as np
from os import path as osp
import scipy.misc as scm
import matplotlib.pyplot as plt

def rec_proto():
	prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1()
	exp         = se.setup_experiment(prms, cPrms)
	dep         = mpu.ProtoDef.recproto_from_proto(exp.expFile_.netDef_, 
									dataLayerNames=['window_data'], newDataLayerNames=['data'],
								  batchSz=10, delLayers=['slice_pair'])
	return dep


def reconstruct():
	prms, cPrms = mepo.smallnetv5_fc5_pose_euler_crp192_rawImSz256_lossl1()
	exp         = se.setup_experiment(prms, cPrms)
	im          = np.ones((10, 101, 101, 3)).astype(np.uint8)
	meanFile    = '/data0/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto'
	kwargs = {}
	kwargs['delAbove']          = 'conv1'
	kwargs['delLayers']         = ['slice_pair'] 
	kwargs['dataLayerNames']    = ['window_data']
	kwargs['newDataLayerNames'] = ['data']
	vu.reconstruct_optimal_input(exp, 20000, im, meanFile=meanFile, batchSz=10, **kwargs)


def get_normals():
	prms, cPrms = mev2.smallnetv5_fc5_nrml_crp192_rawImSz256_nojitter_l1loss()
	wFile = prms.paths['windowFile']['test']
	wFid  = mpio.GenericWindowReader(wFile)
	print (wFid.num_)
	allLb = []
	imDat = []
	for	 i in range(4000):
		imd, lbls = wFid.read_next()
		allLb.append(lbls[0][0:2].reshape((1,2)))
		imDat.append(imd)
	allLb = np.concatenate(allLb, axis=0)
	mag  = np.sum(allLb * allLb, 1)
	idx = np.argsort(mag)
	
	oFile = open('nrml-frontal.txt', 'w')
	for i in idx[0:100]:
		imd = imDat[i][0].strip()
		nrmlStr = '%f \t %f' % (allLb[i][0], allLb[i][1])
		oFile.write('%s \t %s \n' % (imd, nrmlStr))
	oFile.close()	
	#return sMag

def read_normals_fronal(isSave=False, 
					rootFolder='/data0/pulkitag/data_sets/streetview/proc/resize-im/im256'):
	fName = 'nrml-frontal.txt'
	fid   = open(fName, 'r')
	lines = fid.readlines()
	for l in lines:
		imName, ch, w, h, x1, y1, x2, y2, yaw, pitch = l.strip().split()
		yaw, pitch = float(yaw), float(pitch)
		fName = osp.join(rootFolder, imName)
		im    = scm.imread(fName)
		plt.imshow(im)
		if isSave:
			outName = osp.join('debug-data', '%05d.jpg' % count)
			plt.savefig(outName)
		else:	
			inp = raw_input('Press a key to continue')
			if inp=='q':
				return
	
