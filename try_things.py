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
import rot_utils as ru
from scipy import linalg as linalg
import nibabel.quaternions as nq
#from transforms3d.transforms3d import euler as eu
from multiprocessing import Pool, Array
import street_process_data as spd

def get_mat(head, pitch, roll, isRadian=False):
	if not isRadian:
		head  = np.pi * head/180.
		pitch = np.pi * pitch/180.
		roll  = np.pi * roll/180. 
	#mat = ru.euler2mat(head, pitch, roll, isRadian=False)
	mat  = eu.euler2mat(roll, pitch, head, axes='sxyz')
	return mat

def e2q(h, p, r, isRadian=False):
	if not isRadian:
		h = np.pi * h/180.
		p = np.pi * p/180.
		r = np.pi * r/180. 
	q = eu.euler2quat(r, p, h, axes='sxyz')
	return q

def get_mat_dist(m1, m2):
	return linalg.norm(linalg.logm(np.dot(np.transpose(m1), m2)), ord='fro')


def eq():
	head1, pitch1, roll1 = 140, 80, 50
	head2, pitch2, roll2 = 75, 60, 20
	mat1 = get_mat(head1, pitch1, roll1)
	mat2 = get_mat(head2, pitch2, roll2)
	mat  = np.dot(mat1, mat2.transpose())
	h, p, r = ru.mat2euler(mat, seq='xyz')
	print (h,p,r)
	q1 = e2q(head1, pitch1, roll1)
	q2 = e2q(head2, pitch2, roll2)
	q  = nq.mult(q1, q2)
	ang = eu.quat2euler(q, axes='sxyz')
	print(ang)


def try_rotations():
	head1, pitch1, roll1 = 140, 80, 50
	head2, pitch2, roll2 = 75, 60, 20
	mat1 = get_mat(head1, pitch1, roll1)
	mat2 = get_mat(head2, pitch2, roll2)
	mat  = np.dot(mat1, mat2.transpose())
	dMat = get_mat(head1 - head2, pitch1- pitch2, roll1-roll2)
	predMat2 = np.dot(mat1, dMat)
	print (get_mat_dist(mat2, predMat2))
	diff = linalg.norm(linalg.logm(np.dot(np.transpose(mat), dMat)), ord='fro')
	print diff
	h, p, r = ru.mat2euler(mat, seq='xyz')	
	print (h, p, r)
	print (ru.mat2euler(dMat, seq='xyz'))
	return h, p, r

def try_rotations_quat():
	head1, pitch1, roll1 = 140, 40, 0
	head2, pitch2, roll2 = 75, 60, 0
	q1 = ru.euler2quat(head1, pitch1, roll1, isRadian=False)
	q2 = ru.euler2quat(head2, pitch2, roll2, isRadian=False)
	mat1 = nq.quat2mat(q1)
	mat2 = nq.quat2mat(q2)
	q3    = q2
	q3[0] = -q3[0]
	mat  = np.dot(mat2.transpose(), mat1)
	dMat = nq.quat2mat(nq.mult(q3, q1))
	diff = linalg.norm(linalg.logm(np.dot(np.transpose(mat), dMat)), ord='fro')
	print diff
	print (mat - dMat)
	h, p, r = ru.mat2euler(mat)	
	print (h, p, r)
	return h, p, r


def try_rot():
	h1, p1, r1 = 10, 20, 0
	h2, p2, r2 = 20, 10, 0
	mat1  = get_mat(h1, p1, r1)
	mat2  = get_mat(h2, p2, r2)
	mat   = np.dot(mat2, mat1)
	eu    = ru.mat2euler(mat)
	eu    = np.array(eu) * 180/np.pi
	print (eu)

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

def _parallel_group_load(args):
	return True


def try_group_load(data=None):
	if data is None:
		fName = 'tmp/targetGrps.pkl'	
		data  = pickle.load(open(fName, 'r'))
	keys   = data['groups'].keys()
	inArgs = [] 
	for k in keys[0:20]:
		inArgs.append(data['groups'][k])
	pool = Pool(processes=1)
	jobs = pool.map_async(_parallel_group_load, inArgs)
	res  = jobs.get()
	del pool
		
def try_group_load_v2(data=None):
	if data is None:
		fName = 'tmp/targetGrps.pkl'	
		data  = pickle.load(open(fName, 'r'))
	keys   = data['groups'].keys()
	gArr = [data['groups'][k] for k in keys[0:10]] 
	arr  = Array(spd.StreetGroup, gArr)
	inArgs = [] 
	for k in keys[0:20]:
		inArgs.append((k, arr))
	pool = Pool(processes=1)
	jobs = pool.map_async(_parallel_group_load, inArgs)
	res  = jobs.get()
	del pool

