import os
import os.path as osp
import sys
import platform

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print 'added {}'.format(path)

add_path(os.getcwd())

def params():
    config = {}
    config['basedir'] = os.getcwd()
    config['synthDataDir'] = '/home/eecs/shubhtuls/cachedir/blenderRenderPreprocess/'
    config['rotationDataDir'] = osp.join(config['basedir'],'cachedir','rotationDataJoint')
    config['pascalImagesDir'] = osp.join('/','data1','shubhtuls','code','poseInduction','data','VOCdevkit','VOC2012','JPEGImages')
    config['imagenetImagesDir'] = osp.join('/','data1','shubhtuls','code','poseInduction','data','imagenet','images')
    config['pascalTrainValIdsFile'] = osp.join('/','data1','shubhtuls','code','poseInduction','cachedir','pascalTrainValIds.mat')
    return config
