## @package setup_data
#	Functions for setting up the data

import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess
import matplotlib.pyplot as plt
import mydisplay as mydisp
import h5py as h5
import pickle
import street_params as sp
import scipy.misc as scm
from multiprocessing import Pool, Manager, Queue, Process
import time
import copy

##
#Get the list of tar filenames
def get_tar_files(prms):
	with open(prms.paths.tar.fileList,'r') as f:
		fNames = f.readlines()
	fNames = [f.strip() for f in fNames]
	return fNames

##
#Download the tar files from the server
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

##
#Untar the files and then delete them. 
def untar_and_del(prms, isDel=False, fNames=None):
	if fNames is None:
		fNames = get_tar_files(prms)	
	suffix = []
	for f in fNames:
		suffix.append(osp.split(f)[1])
	fNames = [osp.join(prms.paths.tar.dr, s) for s in suffix]
	for f in fNames:
		if not osp.exists(f):
			continue
		subprocess.check_call(['tar -xf %s -C %s' % (f, prms.paths.raw.dr)],shell=True)
		if isDel: 
			subprocess.check_call(['rm %s' % f],shell=True) 
	return fNames	

##
#Helper function used by get_foldernames
def _find_im_labels(folder):
	fNames = os.listdir(folder)
	fNames = [osp.join(folder, f.strip()) for f in fNames]
	#If one is directory then all should be dirs
	print fNames[0]
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
# When the files have been untarred, search for all folders in which
# data has been stored. 
def get_foldernames(prms):
	'''
		Search for the folder tree - till the time there are no more
		directories. We will assume that the terminal folder has all
	  the needed files
	'''
	fNames = [f.strip() for f in os.listdir(prms.paths.raw.dr)]
	fNames = [osp.join(prms.paths.raw.dr, f) for f in fNames]
	allNames = []
	for f in fNames:
		if osp.isdir(f):
			print ('Finding for %s' % f)
			allNames.append(_find_im_labels(f))
	return allNames

##
# Save the foldernames along with the folder ids
def save_foldernames(prms):	
	fNames = sorted(get_foldernames(prms))
	fid    = open(prms.paths.proc.folders.key, 'w')
	allNames = []
	for f in fNames:
		for ff in f:
			allNames  = allNames + [ff]
	allNames = sorted(allNames)
	for (i,f) in enumerate(allNames):
		fid.write('%04d \t %s\n' % (i + 1 ,f))
	fid.close()
	#For safety strip write permissions from the file
	subprocess.check_call(['chmod a-w %s' % prms.paths.proc.folders.key], shell=True) 

##
#If new folders are added, instead of rehashing all folders to new ids, 
#find the new folders and append them to the key store
def append_foldernames(prms):
	keyFile = prms.paths.proc.folders.key
	#Read the Key and folders from the key file
	keys, folders = [], []
	with open(keyFile,'r') as f:
		lines = f.readlines()
		for l in lines:
			key, folder = l.strip().split()
			keys.append(key)
			folders.append(folder)	
	N = len(keys)

	newFolders = []
	fNames = sorted(get_foldernames(prms))
	for f in fNames:
		for ff in f:
			newFolders  = newFolders + [ff]
	newFolders = sorted(newFolders)	

	for nf in newFolders:
		if nf not in folders:
			N = N + 1
			keyStr = '%04d' % N
			keys.append(keyStr)
			folders.append(nf)			
	subprocess.check_call(['chmod a+w %s' % keyFile], shell=True) 
	with open(keyFile, 'w') as fid:
		for k, f in zip(keys, folders):
			fid.write('%s \t %s\n' % (k,f))
	subprocess.check_call(['chmod a-w %s' % keyFile], shell=True) 
	return newFolders, keys, folder	

##
#tar the crop images by folderid
def tar_crop_images_by_folderid(args):
	prms, folderId = args
	drName = prms.paths.proc.im.folder.dr % folderId
	trFile = prms.paths.proc.im.folder.tarFile % folderId
	if not osp.exists(trFile):
		print ('Making %s' % trFile)
		subprocess.check_call(['tar -cf %s %s' % (trFile, drName)],shell=True)
		return True
	else:
		print ('Already exists %s' % trFile)
		return False

##
#Tar the crop images in all folders
#Useful for transferring data across machines
def tar_crop_images(prms):
	folderKeys = su.get_geo_folderids(prms)
	inArgs     = []
	for k in folderKeys:
		inArgs.append([prms, k])	
	pool = Pool(processes=12)
	jobs = pool.map_async(tar_crop_images_by_folderid, inArgs)	
	res  = jobs.get()
	del pool



