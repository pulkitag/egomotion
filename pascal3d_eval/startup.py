import os
import os.path as osp
import sys
import platform
HERE_PATH = os.path.dirname(os.path.realpath(__file__))
import street_config as cfg
from easydict import EasyDict as edict

DATA_DIR = cfg.pths.mainDataDr
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print 'added {}'.format(path)

add_path(os.getcwd())

def params():
	config = edict()
	config['basedir'] = HERE_PATH
	config['cachedir'] = osp.join(config.basedir, 'cachedir')
	config['datadir'] = osp.join(config.basedir, 'datadir')
	config['synthDataDir'] = osp.join(config.cachedir, 'blenderRenderPreprocess/')
	config['rotationDataDir'] = osp.join(config.cachedir,'rotationDataJoint')
	config['pascalImagesDir'] = osp.join(config.datadir, 'VOCdevkit','VOC2012','JPEGImages')
	config['imagenetImagesDir'] = osp.join('/','data1','shubhtuls','code','poseInduction','data','imagenet','images')
	config['pascalTrainValIdsFile'] = osp.join(config.cachedir,'pascalTrainValIds.mat')
	return config
