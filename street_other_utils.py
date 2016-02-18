## @package street_misc_utils
# Miscellaneos utility functions
#

import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess
import matplotlib.pyplot as plt
import mydisplay as mydisp
#import h5py as h5
import pickle
import my_pycaffe_io as mpio
import re
import matplotlib.path as mplPath
import rot_utils as ru
from geopy.distance import vincenty as geodist
import copy
import street_params as sp
from multiprocessing import Pool


##
#Fetch the window file from the main machine
def fetch_window_file_scp(prms):
	setNames = ['train', 'test']
	hostName = 'ubuntu@54.173.41.3:/data0/pulkitag/data_sets/streetview/exp/window-files/'
	for s in setNames:
		wName      = prms['paths']['windowFile'][s]
		_, fName   = osp.split(wName)
		remoteName = hostName + fName
		scpCmd = 'scp -i "pulkit-key.pem" '
		localName = prms['paths']['windowFile'][s]
		subprocess.check_call(['%s %s %s' % (scpCmd, remoteName, localName)],shell=True) 

##
#Fetch the croppped images
def fetch_cropim_tar_by_folderid(args):
	prms, folderId = args
	hostName = 'ubuntu@54.173.41.3:/data0/pulkitag/data_sets/streetview/proc/resize-im/im256/'
	trFile = prms.paths.proc.im.folder.tarFile % folderId
	remoteName = hostName + '%s.tar' % folderId
	scpCmd = 'scp -i "pulkit-key.pem" '
	localName = trFile
	subprocess.check_call(['%s %s %s' % (scpCmd, remoteName, localName)],shell=True) 


##
#Convert a pose patch window file into a window file for patch matching only
def convert_pose_ptch_2_ptch(inFile, outFile):
	inFid  = mpio.GenericWindowReader(inFile)
	outFid = mpio.GenericWindowWriter(outFile, inFid.num_, 2, 1)
	while not inFid.is_eof():
		imData, lb = inFid.read_next()
		lbls = [[lb[0][2]]]	
		outFid.write(lbls[0], *imData)
	inFid.close()
	outFid.close()


#Send the window file to a host
def send_window_file_scp(prms, setNames=None):
	if setNames is None:
		setNames = ['train', 'test']
	hostName = 'pulkitag@psi.millennium.berkeley.edu:/work5/pulkitag/data_sets/streetview/'
	for s in setNames:
		wName      = prms['paths']['windowFile'][s]
		_, fName   = osp.split(wName)
		remoteName = hostName + fName
		scpCmd = 'scp '
		localName = prms['paths']['windowFile'][s]
		subprocess.check_call(['%s %s %s' % (scpCmd, localName, remoteName)],shell=True) 








'''
##
#Process the labels according to prms
def get_labels_old(prms, setName="train"):
	#The main quantity that requires randomization is patch matching
	#So we will base this code around that. 
	rawLb = get_groups_all(prms, setName=setName)
	N  = len(rawLb)
	oldState  = np.random.get_state()
	randSeed  = 1001
	randState = np.random.RandomState(randSeed)
	perm1     = randState.permutation(N)
	perm2     = randState.permutation(N)	
	perms     = zip(perm1,perm2)
	#get the labels
	lb, prefix = [], []
	for (i, perm) in enumerate(perms):
		p1, p2 = perm
		for lbType in prms.labels:
			if lbType.label_ == 'nrml':
				#1 because we are going to have this as input to the
				# ignore euclidean loss layer
				rl = rawLb[p1]
				for i in range(rl.num):
					#Ignore the last dimension as its always 0.
					lb.append(rl.data[i].nrml[0:2])
					prefix.append((rl.folderId, rl.prefix[i].strip(), None, None))
			elif lbType.label_ == 'ptch':
				#Based on the idea that there are typically 5 samples per group
				numRep = int(5.0 / lbType.posFrac_)
				for rep in range(numRep):
					prob   = randState.rand()
					p1, p2 = randState.random_integers(N-1), randState.random_integers(N-1)
					rl1  = rawLb[p1]
					rl2  = rawLb[p2]
					localPerm1 = randState.permutation(rl1.num)
					localPerm2 = randState.permutation(rl2.num)
					if prob > lbType.posFrac_:
						#Sample positive
						lb.append([1])	
						prefix.append((rl1.folderId, rl1.prefix[localPerm1[0]].strip(),
													 rl1.folderId, rl1.prefix[localPerm1[1]].strip()))
					else:
						#Sample negative			
						lb.append([0])
						prefix.append((rl1.folderId, rl1.prefix[localPerm1[0]].strip(),
													 rl2.folderId, rl2.prefix[localPerm2[0]].strip()))
			elif lbType.label_ == 'pose':
				rl1        = rawLb[p1]
				for n1 in range(rl1.num):
					for n2 in range(n1+1, rl1.num):
						if rl1.data[n1].align is None or rl1.data[n2].align is None:
							continue	 
						y1, x1, z1 = rl1.data[n1].rots
						y2, x2, z2 = rl1.data[n2].rots
						roll, yaw, pitch = z2 - z1, y2 - y1, x2 - x1
						if lbType.maxRot_ is not None:
							if (np.abs(roll) > lbType.maxRot_ or\
								  np.abs(yaw) > lbType.maxRot_ or\
								  np.abs(pitch)>lbType.maxRot_):
									continue
						if lbType.labelType_ == 'euler':
							if lbType.lbSz_ == 3:
								lb.append([roll/180.0, yaw/180.0, pitch/180.0]) 
							else:
								lb.append([yaw/180.0, pitch/180.0]) 
						elif lbType.labelType_ == 'quat':
							quat = ru.euler2quat(z2-z1, y2-y1, x2-x1, isRadian=False)
							q1, q2, q3, q4 = quat
							lb.append([q1, q2, q3, q4]) 
						else:
							raise Exception('Type not recognized')	
						prefix.append((rl1.folderId, rl1.prefix[n1].strip(),
													 rl1.folderId, rl1.prefix[n2].strip()))
			else:
				raise Exception('Type not recognized')	
	np.random.set_state(oldState)		
	return lb, prefix					

'''
