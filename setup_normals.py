import scipy.io as sio
import numpy as np
import street_config as cfg
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import os
from os import path as osp
import other_utils as ou
import scipy.misc as scm

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

def get_paths():
	expDir, dataDir = cfg.pths.nyu.expDr, cfg.pths.nyu.dataDr
	pth = edict()	
	pth.exp = edict()
	pth.exp.dr= expDir
	#Snapshots
	pth.exp.snapshot    = edict()
	pth.exp.snapshot.dr = osp.join(pth.exp.dr, 'snapshot')
	ou.mkdir(pth.exp.snapshot.dr)
	#Get the label-stats
	pth.exp.labelStats = osp.join(pth.exp.dr, 'label_stats.pkl')
	#info label for the experiment
	#pth.exp.lbInfo     = osp.join(pth.exp.dr, 'label_info', dPrms.expStr, 'lbinfo.pkl') 
	#ou.mkdir(osp.dirname(pth.exp.lbInfo))
	#Results
	pth.exp.results = edict()
	pth.exp.results.dr   = osp.join(pth.exp.dr, 'results', '%s')
	pth.exp.results.file = osp.join(pth.exp.results.dr, 'iter%d.pkl') 
	#Data files
	pth.data      = edict()
	pth.data.dr   = dataDir	
	pth.data.gtnrmlRaw = osp.join(dataDir, 'normals_gt', 'normals','%04d.png')
	pth.data.gtnrml    = osp.join(dataDir, 'normals_gt_renamed', 'normals', '%04d.png')
	pth.data.imRaw     = osp.join(dataDir, 'ims', 'im%04d.jpg')
	pth.data.imFolder = osp.join(dataDir, 'imCrop', 'imSz%d_pad%d')
	#pth.data.imFolder = pth.data.imFolder % (dPrms.imCutSz, dPrms.imPadSz)
	#base net files
	pth.baseProto = osp.join(REAL_PATH, 'base_files', '%s.prototxt')
	#Window files
	windowDr      = osp.join(REAL_PATH, 'pose-files')
	pth.window  = edict()
	#Window files stores theta in degrees
	#pth.window.train = osp.join(windowDr, 'euler_train_pascal3d_imSz%d_pdSz%d.txt')
	#pth.window.test  = osp.join(windowDr, 'euler_test_pascal3d_imSz%d_pdSz%d.txt')
	#pth.window.train = pth.window.train % (dPrms.imCutSz, dPrms.imPadSz)
	#pth.window.test  = pth.window.test %  (dPrms.imCutSz, dPrms.imPadSz)
	return pth	

#original files are named starting with 0, rename them to start with 1
def rename_gt_normals(pths=None):
	if pths is None:
		pths = get_paths()
	for i in range(0,1449):
		rawName = pths.data.gtnrmlRaw % i
		finName = pths.data.gtnrml  % (i+1)		 
		subprocess.check_call(['cp %s %s' % (rawName, finName)], shell=True)

def show_images(pths=None):
	plt.ion()
	if pths is None:
		pths = get_paths()
	for i in range(100):
		im = scm.imread(pths.data.imRaw % (i+1))
		plt.imshow(im)		
		plt.show()
		plt.draw()
		ip = raw_input()
		if ip == 'q':
			return			

def blah():
	pass
	

def compute_normal_centers(nrmls):
	'''
		nrmls: N x 3 where N is the number of points
	'''

def read_normal_file(fName):
	dat = sio.loadmat(sio)
	
