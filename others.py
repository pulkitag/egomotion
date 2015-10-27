import scipy.misc as scm
import cv2
import my_pycaffe_io as mpio
import street_exp as se
from os import path as osp
import time

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
		
		
