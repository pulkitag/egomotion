import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess

def process_folder():
	pass

def get_foldernames(prms):
	'''
		Search for the folder tree - till the time there are no more
		directories. We will assume that the terminal folder has all
	  the needed files
	'''
	fNames = osp.listdir(prms.paths.rawData)
	fNames = [f for f in fNames if osp.isdir(f)]
	


def get_tar_files(prms):
	with open(prms.paths.tarListFile,'r') as f:
		fNames = f.readlines()
	fNames = [f.strip() for f in fNames]
	return fNames

def download_tar(prms):
	fNames = get_tar_files(prms)
	for f in fNames:
		_, name = osp.split(f)
		outName = osp.join(prms.paths.dirs.tarData, name)
		print(outName)
		if not osp.exists(outName):
			print ('Downloading')
			subprocess.check_call(['gsutil cp %s %s' % (f, outName)], shell=True) 
	print ('All Files copied')
	

def get_paths():
	paths      = edict()
	#For storing the directories
	paths.dirs = edict()
	#The raw data
	paths.dirs.data     = '/data1/pulkitag/data_sets/streetview'
	paths.dirs.rawData  = osp.join(paths.dirs.data, 'raw')
	paths.dirs.procData = osp.join(paths.dirs.data, 'proc')
	paths.dirs.tarData  = osp.join(paths.dirs.data, 'tar')
	paths.dirs.code     = '/home/ubuntu/code/streetview'
	paths.tarListFile   = osp.join(paths.dirs.code, 'data_list.txt') 
	return paths


def get_data_files(prms):
	allNames = os.listdir(prms.paths.dirs.rawData)
	#Extract the prefixes
	imNames   = sorted([f for f in allNames if '.jpg' in f], reverse=True)
	lbNames   = sorted([f for f in allNames if '.txt' in f], reverse=True)
	prefixStr = []
	for (i,imn) in enumerate(imNames):
		imn = imn[0:-4]
		if imn in lbNames[i]:
			prefixStr = prefixStr + [imn] 
	imNames = [f + '.jpg' for f in prefixStr]
	lbNames = [f + '.txt' for f in prefixStr]
	return imNames, lbNames
	 


def get_prms():
	prms = edict()
	prms.paths = get_paths()
	return prms

