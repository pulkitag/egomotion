import socket
import os
from os import path as osp
from easydict import EasyDict as edict

pths = edict()
REAL_PATH = os.path.dirname(os.path.realpath(__file__))
HOST_NAME = socket.gethostname()
DEF_DB    = osp.join(REAL_PATH, 'exp-data/db-store/%s-%s-%s-db.sqlite')
if 'ivb' in HOST_NAME:
	HOST_STR = 'nvCluster'
else:
	pths.mainDataDr = '/data0/pulkitag/data_sets/streetview'
	pths.expDir     = '/data0/pulkitag/streetview/exp'
	HOST_STR = HOST_NAME
DEF_DB    = DEF_DB % ('%s',HOST_STR, '%s')

#Other paths
pths.folderProc = osp.join(pths.mainDataDr, 'proc2', '%s')
pths.cwd = osp.dirname(osp.abspath(__file__))

