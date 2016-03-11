## @package scp_utils
# Functions involved in file transfers
#

import subprocess
import os
import os.path as osp

def get_hostaddr(hostName):
	if hostName == 'psi':
		addr   = 'pulkitag@psi.millennium.berkeley.edu:/work5/pulkitag/data_sets/streetview/'
	elif hostName == 'nvidia':
		addr  = 'pagrawal@psglogin.nvidia.com:/puresan/shared/pulkitag/data_sets/streetview/'
	elif hostName == 'anakin':
		addr  = 'pulkitag@anakin.banatao.berkeley.edu:'
	elif hostName == 'server':
		addr  = '-i "pulkit-key.pem" ubuntu@52.91.22.126:'
	else:
		raise Exception('Not found %s' % hostName)
	return addr


def transfer_snapshot(exp, numIter, targetName=None):
	snapName = exp.get_snapshot_name(numIter)
	if not osp.exists(snapName):
		print ('%s doenot exists' % snapName)
		return
	hostAddr = get_hostaddr('anakin')
	trFile   = hostAddr + '/work4/pulkitag-code/code/projStreetView/exp-data/snaps'
	if targetName is not None:
		name = targetName + ('_%d' % numIter) + '.caffemodel'
		trFile = osp.join(trFile, name) 
	subprocess.check_call(['scp %s %s' % (snapName, trFile)], shell=True)

##
#Helper for scp_cropim_tars
def scp_cropim_tar_by_folderid(args):
	prms, folderId, hostPath = args
	trFile = prms.paths.proc.im.folder.tarFile % folderId
	print ('Transferring %s' % trFile)
	subprocess.check_call(['scp %s %s' % (trFile, hostPath)],shell=True)

##
#Send the cropped image files for all folders to hostname
def scp_cropim_tars(prms, hostName='psi'):
	folderKeys = su.get_geo_folderids(prms)
	if hostName == 'psi':
		hostAddr   = osp.join(get_hostaddr(hostName), 'resize-im', 'im256') 
	else:
		hostAddr   = osp.join(get_hostaddr(hostName), 'proc', 'resize-im', 'im256') 
	inArgs     = []
	for k in folderKeys:
		inArgs.append([prms, k, hostAddr])	
	pool = Pool(processes=6)
	jobs = pool.map_async(scp_cropim_tar_by_folderid, inArgs)	
	res  = jobs.get()
	del pool


##
#Fetch the window file from server
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
#Fetch croppped images from server
def fetch_cropim_tar_by_folderid(args):
	prms, folderId = args
	hostName = 'ubuntu@54.173.41.3:/data0/pulkitag/data_sets/streetview/proc/resize-im/im256/'
	trFile = prms.paths.proc.im.folder.tarFile % folderId
	remoteName = hostName + '%s.tar' % folderId
	scpCmd = 'scp -i "pulkit-key.pem" '
	localName = trFile
	subprocess.check_call(['%s %s %s' % (scpCmd, remoteName, localName)],shell=True) 


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


