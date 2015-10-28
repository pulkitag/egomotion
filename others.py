import scipy.misc as scm
import cv2
import my_pycaffe_io as mpio
import street_exp as se
from os import path as osp
import time
import street_utils as su
import my_exp_v2 as mev2

def benchmark_reads(prms, setName='train'):
	wFile = prms.paths.windowFile[setName]
	wFid  = mpio.GenericWindowReader(wFile)
	rootDir = se.get_windowfile_rootdir(prms)
	allNames = []
	for i in range(2000):
		imName, lbls = wFid.read_next()
		allNames.append(osp.join(rootDir,imName[0].strip().split()[0]))
	
	t1 = time.time()
	for i in range(2000):
		im = cv2.imread(allNames[i])
	t2 = time.time()
	print ('TIme Elapsed: %f' % (t2-t1))
	return allNames
	

def make_window_files(prms):	
	su.make_window_files_geo_folders(prms)	
	su.make_combined_window_file(prms, 'train')
	su.make_combined_window_file(prms, 'test')

def make_window_files_many():
	#Pose Max 45
	print ('Max Euler 45')
	prms, cPrms = mev2.smallnetv2_pool4_pose_euler_mx45_crp192_rawImSz256()
	make_window_files(prms)
	print ('Max Euler 90')
	prms, cPrms = mev2.smallnetv2_pool4_pose_euler_mx90_crp192_rawImSz256()
	make_window_files(prms)
	print ('Patch Files')
	prms, cPrms = mev2.smallnetv2_pool4_ptch_crp192_rawImSz256()
	make_window_files(prms)
	print ('Pose Patch Files')
	prms, cPrms = mev2.ptch_pose_euler_mx45_exp2()	
	make_window_files(prms)
