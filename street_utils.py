import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess

def _mkdir(fName):
	if not osp.exists(fName):
		os.makedirs(fName)

def process_folder():
	pass

##
# Helper function for get_foldernames
def _find_im_labels(folder):
	fNames = os.listdir(folder)
	fNames = [osp.join(folder, f.strip()) for f in fNames]
	#If one is directory then all should be dirs
	if osp.isdir(fNames[0]):
		dirNames = []
		for f in fNames:
			childDir = _find_im_labels(f)	
			if type(childDir) == list:
				dirNames = dirNames + childDir
			else:
				dirNames = dirNames + [childDir]
	else:
		dirNames = [folder]
	print dirNames
	return dirNames

##
# Find the foldernames in which data is stored
def get_foldernames(prms):
	'''
		Search for the folder tree - till the time there are no more
		directories. We will assume that the terminal folder has all
	  the needed files
	'''
	fNames = [f.strip() for f in os.listdir(prms.paths.raw.dr)]
	fNames = [osp.join(prms.paths.raw.dr, f) for f in fNames]
	print (fNames)
	fNames = [_find_im_labels(f) for f in fNames if osp.isdir(f)]
	return fNames

##
# Save the foldernames along with the folder ids
def save_foldernames(prms):	
	fNames = sorted(get_foldernames(prms))
	fid    = open(prms.paths.proc.folders.key, 'w')
	count  = 0
	for (i,f) in enumerate(fNames):
		if len(f) > 1:
			for ff in f:
				fid.write('%04d \t %s\n' % (i + 1 + count,ff))
				count += 1
		else:
			fid.write('%04d \t %s\n' % (i + 1 + count,f[0]))
	fid.close()

##
# Read names of all files in the folder
# Ensure that .jpg and .txt match and save those prefixes
def read_prefixes_from_folder(dirName):
	allNames = os.listdir(dirName)
	#Extract the prefixes
	imNames   = sorted([f for f in allNames if '.jpg' in f], reverse=True)
	lbNames   = sorted([f for f in allNames if '.txt' in f], reverse=True)
	prefixStr = []
	for (i,imn) in enumerate(imNames):
		imn = imn[0:-4]
		if i>= len(lbNames):
			continue
		if imn in lbNames[i]:
			prefixStr = prefixStr + [imn] 
	return prefixStr

##
# Save filename prefixes for each folder
def save_folder_prefixes(prms):
	fid   = open(prms.paths.proc.folders.key, 'r')
	lines = [l.strip() for l in fid.readlines()]
	fid.close()
	keys, names = [], []
	for l in lines:
		key, name = l.split()
		print (key, name)
		fName   = prms.paths.proc.folders.pre % key
		if osp.exists(fName):
			continue
		preStrs = read_prefixes_from_folder(name) 
		with open(fName, 'w') as f:
			for p in preStrs:
				f.write('%s \n' % p)
		

def get_tar_files(prms):
	with open(prms.paths.tar.fileList,'r') as f:
		fNames = f.readlines()
	fNames = [f.strip() for f in fNames]
	return fNames

def download_tar(prms):
	fNames = get_tar_files(prms)
	for f in fNames:
		_, name = osp.split(f)
		outName = osp.join(prms.paths.tar.dr, name)
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
	paths.dataDr  = '/data0/pulkitag/data_sets/streetview'
	paths.raw     = edict()
	paths.raw.dr  = osp.join(paths.dataDr, 'raw')
	#Processed data
	paths.proc    = edict()
	paths.proc.dr = osp.join(paths.dataDr, 'proc')
	#Raw Tar Files
	paths.tar     = edict()
	paths.tar.dr  = osp.join(paths.dataDr, 'tar')
	#The Code directory
	paths.code    = edict()
	paths.code.dr = '/home/ubuntu/code/streetview'
	#List of tar files
	paths.tar.fileList = osp.join(paths.code.dr, 'data_list.txt') 
	#Store the names of all the folders
	paths.proc.folders    = edict()
	paths.proc.folders.dr = osp.join(paths.proc.dr, 'folders')
	_mkdir(paths.proc.folders.dr)
	#Stores the folder names along with the keys
	paths.proc.folders.key  = osp.join(paths.proc.folders.dr, 'key.txt') 
	paths.proc.folders.pre  = osp.join(paths.proc.folders.dr, '%s.txt') 
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


def parse_label_file(fName):
	label = edict()
	with open(fName, 'r') as f:
		data = f.readlines()
		dl   = data[0].strip().split()
		assert dl[0] == 'd'
		label.ids    = edict()
		label.ids.ds = int(dl[1]) #DataSet id
		label.ids.tg = int(dl[2]) #Target id
		label.ids.im = int(dl[3]) #Image id
		label.ids.sv = int(dl[4]) #Street view id
		#dl[5,6,7] -- target point will skip
		label.nrml   = [float(n) for n in dl[8:11]]
		#dl[11,12,13] -- street view point not needed
		label.dist   = float(dl[14])
		label.rots   = [float(n) for n in dl[15:18]]
		assert label.rots[2] == 0, 'Something is wrong %s' % fName	
		label.align = None
		if len(data) == 2:
			al = data[0].strip().split()
			label.align = edict()
			#Corrected patch center
			label.align.loc	 = [float(n) for n in al[0:2]]
			#Warp matrix	
			label.align.warp = np.array([float(n) for n in al[2:11]])
	return label
			
