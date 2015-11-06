import scipy.misc as scm
import cv2
import my_pycaffe_io as mpio
import street_exp as se
from os import path as osp
import time
import street_utils as su
import my_exp_v2 as mev2
import street_params as sp
import numpy as np

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


def hack_window_file():
	targetDir = '/data0/pulkitag/hack'
	inDir     = '/data0/pulkitag/data_sets/streetview/proc/resize-im/im256/'
	prms = sp.get_prms_vegas_ptch()
	wFile = mpio.GenericWindowReader(prms.paths['windowFile']['test'])
	readFlag = True
	count    = 0
	while readFlag:
		if np.mod(count,1000)==0:
			print (count)
		count += 1
		ims, lb = wFile.read_next()
		for im in ims:
			fName = im.strip().split()[0]
			inName  = osp.join(inDir, fName)
			inDat   = scm.imread(inName)
			outName = osp.join(targetDir, fName)
			dirName, _ = osp.split(outName)
			sp._mkdir(dirName)
			scm.imsave(outName, inDat) 
		if wFile.is_eof():
			readFlag = False


def window_file_ptch_gt_euler_5():
	prms, cPrms = mev2.ptch_pose_euler_smallnet_v5_fc5_exp1_lossl1()
	wTest = prms.paths['windowFile']['test']
	wFid  = mpio.GenericWindowReader(wTest)
	oName = 'ptch_test_euler-gt5.txt'
	readFlag = True
	count    = 0
	imDat, lbDat = [], []
	while readFlag:
		ims, lb = wFid.read_next()
		if lb is None:
			readFlag = False
			continue
		poseLb  = lb[0][0:2]
		mxTheta = 30 * max(np.abs(poseLb))
		if mxTheta < 5 and not(lb[0][3]==0):
			continue
		count += 1
		imDat.append(ims)
		lbDat.append([lb[0][3]])
		if wFid.is_eof():
			readFlag = False
	wFid.close()
	#Outside id
	oFid = mpio.GenericWindowWriter(oName, count, 2, 1)
	for c in range(count):
		oFid.write(lbDat[c], *imDat[c])
	oFid.close()
