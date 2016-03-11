import numpy as np
import sys, os
import os.path as osp
HERE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(osp.dirname(HERE_PATH))
import startup
import scipy.io as sio
import itertools

def flatten(l):
    return list(itertools.chain.from_iterable(l))

def getDatasetImgDir(dataset,config):
    if(dataset == 'pascal'):
        return config['pascalImagesDir'],'.jpg'
    elif(dataset == 'imagenet'):
        return config['imagenetImagesDir'],'.JPEG'

def imgPath(imgName,dataset,config):
    imgDir, imgExt = getDatasetImgDir(dataset,config)
    return os.path.join(imgDir,imgName+imgExt)

def loadAnnos(cName,isTest,useOccluded=0):
    config = startup.params()
    annoFile = os.path.join(config['rotationDataDir'],cName + '.mat')
    var = sio.loadmat(annoFile)
    rotationData = var['rotationData'][0]
    trainValIds = sio.loadmat(config['pascalTrainValIdsFile'])
    fieldNames = np.array(list(rotationData.dtype.names))

    boxId = np.where(fieldNames=='bbox')[0][0]
    imgId = np.where(fieldNames=='voc_image_id')[0][0]
    #recId = np.where(fieldNames=='voc_rec_id')[0][0]
    eulerId = np.where(fieldNames=='euler')[0][0]
    datasetId = np.where(fieldNames=='dataset')[0][0]
    occlusionId = np.where(fieldNames=='occluded')[0][0]

    bboxes = [rd[boxId][0] for rd in rotationData]
    eulers = [flatten(rd[eulerId]) for rd in rotationData]
    datasetNames = [rd[datasetId][0] for rd in rotationData]
    imgNames = [rd[imgId][0] for rd in rotationData]
    occluded = [rd[occlusionId][0][0] for rd in rotationData]

    valIds = [ flatten(x)[0] for x in trainValIds['valIds']]
    classValIds = [ix for ix in range(len(bboxes)) if (imgNames[ix] in valIds)]
    classTrainIds = [ix for ix in range(len(bboxes)) if (ix not in classValIds)]
    selectionInds = classValIds if isTest else classTrainIds
    if useOccluded == -1:
        selectionInds = [ix for ix in selectionInds if occluded[ix] == 1]
    elif useOccluded == 0:
        selectionInds = [ix for ix in selectionInds if occluded[ix] == 0]

    bboxes = [bboxes[ix] for ix in selectionInds]
    eulers = [eulers[ix] for ix in selectionInds]
    datasetNames = [datasetNames[ix] for ix in selectionInds]
    imgNames = [imgNames[ix] for ix in selectionInds]

    imgPaths = [imgPath(imgNames[ix], datasetNames[ix],config) for ix in range(len(imgNames))]
    return imgPaths, bboxes, eulers
