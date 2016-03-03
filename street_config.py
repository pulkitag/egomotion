import socket
import os
from os import path as osp
from easydict import EasyDict as edict

REAL_PATH = os.path.dirname(os.path.realpath(__file__))
HOST_NAME = socket.gethostname()

def get_paths(hostName=None):
	DEF_DB    = osp.join(REAL_PATH, 'exp-data/db-store/%s-%s-%s-db.sqlite')
	pths = edict()
	if hostName is  None:
		hostName = HOST_NAME
	if 'ivb' in hostName:
		HOST_STR = 'nvCluster'
		pths.mainDataDr = '/scratch/pulkitag/data_sets/streetview'
		pths.expDir     = '/scratch/pulkitag/streetview/exp'
	else:
		pths.mainDataDr = '/data0/pulkitag/data_sets/streetview'
		pths.expDir     = '/data0/pulkitag/streetview/exp'
		HOST_STR = hostName
	DEF_DB    = DEF_DB % ('%s',HOST_STR, '%s')

	#Other paths
	pths.folderDerivDir = osp.join(pths.mainDataDr, 'proc2-deriv', '%s', '%s')
	pths.folderDerivDirTar = osp.join(pths.mainDataDr, 'proc2-deriv-tar', '%s', '%s')
	pths.folderProc    = osp.join(pths.mainDataDr, 'proc2', '%s')
	pths.folderProcTar = osp.join(pths.mainDataDr, 'proc2-tar', '%s.tar')
	pths.cwd = osp.dirname(osp.abspath(__file__))
	return pths, DEF_DB

pths, DEF_DB = get_paths()
