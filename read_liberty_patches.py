import scipy.misc as scm
from easydict import EasyDict as edict
from os import path as osp
import cPickle as pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import my_pycaffe_io as mpio
import pdb
from pycaffe_config import cfg

def _mkdir(fName):
	dirName, _ = osp.split(fName)
	if not osp.exists(dirName):
		os.makedirs(dirName)

#Read the positives and negatives from the test
def read_test(prms):
	fName =  prms.paths.testFile
	with open(fName,'r') as fid:
		lines = fid.readlines()
		posMatch = []
		negMatch = []
		for l in lines:
			#print l	
			ptchId1, ptId1, _, ptchId2, ptId2, _, _ = l.strip().split()
			if ptId1 == ptId2:
				posMatch.append([ptchId1, ptId1, ptchId2, ptId2])
			else:
				negMatch.append([ptchId1, ptId1, ptchId2, ptId2])
	return posMatch, negMatch


def get_prms():
	prms = edict()
	prms.paths = edict()
	dataDir    = '/work5/pulkitag/data_sets/patch-match'
	prms.paths.testFile  = osp.join(dataDir, 'liberty/m50_100000_100000_0.txt')
	prms.paths.infoFile  = osp.join(dataDir, 'liberty/info.txt')
	prms.paths.rawIms    = osp.join(dataDir, 'liberty/patches%04d.bmp')
	prms.paths.imData    = osp.join(dataDir, 'proc/liberty/ims/folder-%05d/data-%04d.pkl')
	prms.paths.pt2imHash = osp.join(dataDir, 'proc/liberty/ims/pt2im.pkl') 

	#The imData Dir for saving pos-neg images
	prms.paths.jpgDir = osp.join(cfg.STREETVIEW_DATA_MAIN, 
											'pulkitag/data_sets/patch-match/liberty/proc/imJpgs/')
	prms.paths.wFile  = osp.join(cfg.STREETVIEW_DATA_MAIN, 
											'pulkitag/data_sets/patch-match/liberty/proc/window-file.txt')
	prms.paths.jpgPosKey = osp.join(prms.paths.jpgDir, 'jpg-pos-key.pkl')
	prms.paths.jpgNegKey = osp.join(prms.paths.jpgDir, 'jpg-neg-key.pkl')
	return prms


def pointid_2_iminfo(prms):
	'''
		Given the point id find the images and the location in the images
		that the points 
	'''
	fName   = prms.paths.infoFile
	outName = prms.paths.pt2imHash 
	with open(fName, 'r') as fid:
		lines = fid.readlines()
		pointIds = [l.strip().split()[0] for l in lines]

	prevId      = None
	prevImNum   = None
	ptCount     = 0
	ptHash   = edict()
	keyStr      = '%s'
	folderCount = 0
	svStr       = prms.paths.imData
	imStr       = prms.paths.rawIms 
	imList      = []
	for i,p in enumerate(pointIds):
		if (not (p == prevId) and prevId is not None) or i ==len(pointIds)-1 :
			key     = keyStr % prevId
			svName  = svStr % (folderCount, ptCount)
			ptHash[key] = svName
			_mkdir(svName)
			#print (prevId, svName)
			ims     = np.concatenate(imList, axis=0).astype(np.uint8)
			pickle.dump({'ims': ims}, open(svName, 'w'))
			imList  = []
			prevId  = p
			ptCount += 1
			if ptCount >= 1000:
				ptCount      = 0
				folderCount += 1
				print (folderCount)

		if prevId is None:
			prevId = p	

		imNum = int(np.floor(i / (16 * 16)))
		row   = int(np.floor((i % 256) / 16))
		col   = i % 16
		#print (imNum, row, col)
		if imNum > prevImNum:
			imName = imStr % imNum
			im     = scm.imread(imName)
		rSt = row * 64
		rEn = rSt + 64
		cSt = col * 64
		cEn = cSt + 64
		imList.append(im[rSt:rEn,cSt:cEn].reshape(1,64,64))
	pickle.dump({'pt2im': ptHash}, open(outName,'w'))


def vis_pos_neg():
	prms     = get_prms()
	pos, neg = read_test(prms)
	egs      = pos + neg
	labels   = np.ones((len(egs),))
	labels[len(pos):] = 0
	perm     = np.random.permutation(len(egs))
	egs      = [egs[p] for p in perm]
	labels   = [labels[p] for p in perm]
	fig      = plt.figure()
	plt.ion()
	#Get the key file
	keyFile  = prms.paths.pt2imHash 
	pt2im    = pickle.load(open(keyFile, 'r'))['pt2im']	
	plt.set_cmap('gray')	
	for eg,lb in zip(egs, labels):
		_,p1,_,p2 = eg
		k1       = pt2im['%s' % p1]
		k2       = pt2im['%s' % p2]
		ims1     = pickle.load(open(k1,'r'))['ims']
		ims2     = pickle.load(open(k2,'r'))['ims']
		l1, l2   = ims1.shape[0], ims2.shape[0]
		if k1 ==k2:
			idx1, idx2 = 0, 1
		else:
			idx1 = np.random.permutation(l1)[0]
			idx2 = np.random.permutation(l2)[0]
	
		ax1 = plt.subplot(121)
		ax1.imshow(ims1[idx1])
		ax2 = plt.subplot(122)
		ax2.imshow(ims2[idx2])
		plt.title('Label: %d' % lb)
		ip = raw_input()
		if ip=='q':
			return 

def save_images(prms, keyFile, egs, imCount=0):
	pt2im    = pickle.load(open(prms.paths.pt2imHash, 'r'))['pt2im']	
	imNames  = []
	jpgPath  = 'folder-%05d/im%03d.jpg'
	for i,eg in enumerate(egs):
		try:
			if np.mod(i,1000)==1:
				print (i)
			_,p1,_,p2 = eg
			k1       = pt2im['%s' % p1]
			k2       = pt2im['%s' % p2]
			ims1     = pickle.load(open(k1,'r'))['ims']
			ims2     = pickle.load(open(k2,'r'))['ims']
		except:
			print(k1, k2)
			pdb.set_trace()
		l1, l2   = ims1.shape[0], ims2.shape[0]
		if k1 ==k2:
			idx1, idx2 = 0, 1
		else:
			idx1 = np.random.permutation(l1)[0]
			idx2 = np.random.permutation(l2)[0]
		im1 = ims1[idx1]
		im2 = ims2[idx2]
		im1 = np.tile(im1,(3,1,1)).transpose((1,2,0))
		im2 = np.tile(im2,(3,1,1)).transpose((1,2,0))
		#Save First Image
		folderCount = int(np.floor(imCount/1000))
		fName1 = jpgPath % (folderCount, np.mod(imCount,1000))
		imCount += 1
		svName = osp.join(prms.paths.jpgDir, fName1)
		_mkdir(svName)
		scm.imsave(svName, im1)
		#Save Second Image	
		folderCount = int(np.floor(imCount/1000))
		fName2 = jpgPath % (folderCount, np.mod(imCount,1000))
		imCount += 1
		svName = osp.join(prms.paths.jpgDir, fName2)
		_mkdir(svName)
		scm.imsave(svName, im2)
		imNames.append([fName1, fName2])	
	pickle.dump({'imNames':imNames}, open(keyFile, 'w'))
	return imCount

def _save_images(args):
	save_images(*args)
	return True

def save_pos_neg_images():
	pool = Pool(processes=2)
	prms = get_prms()
	pos, neg = read_test(prms)
	pos = pos[0:45000]
	neg = neg[0:45000]
	inArgs = []
	inArgs.append([prms, prms.paths.jpgPosKey, pos, 0])
	inArgs.append([prms, prms.paths.jpgNegKey, neg, 2 * len(pos)])
	res = pool.map_async(_save_images, inArgs)
	pool.close()
	resData = res.get()
	pool.terminate()

	#save_images(prms, prms.paths.jpgPosKey, pos)
	#save_images(prms, prms.paths.jpgNegKey, neg, imCount=len(pos))

def make_window_file():	
	prms   = get_prms()
	posDat = pickle.load(open(prms.paths.jpgPosKey,'r'))['imNames'] 
	negDat = pickle.load(open(prms.paths.jpgNegKey,'r'))['imNames']
	allDat = posDat + negDat
	labels   = np.ones((len(allDat),))
	labels[len(posDat):] = 0
	perm     = np.random.permutation(len(allDat))
	allDat   = [allDat[p] for p in perm]
	labels   = [labels[p] for p in perm]
	N        = len(allDat)

	wFid   = mpio.GenericWindowWriter(prms.paths.wFile,N,2,1)
	for dat, lb in zip(allDat, labels):
		line = []
		line.append([dat[0], [3, 64, 64], [0, 0, 64, 64]])
		line.append([dat[1], [3, 64, 64], [0, 0, 64, 64]])
		wFid.write([lb], *line)
	wFid.close()
		
		 
	
			
