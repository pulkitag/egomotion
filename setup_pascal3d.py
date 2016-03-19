import numpy as np
import my_pycaffe_io as mpio
#import street_utils as su
import my_exp_v2 as mev2
import street_exp as se
from easydict import EasyDict as edict
from os import path as osp
import cv2
import scipy.misc as scm
import other_utils as ou
import copy
import matplotlib.pyplot as plt
from skimage import color
import math
import pickle
import pdb

def create_window_file():
	setName = ['test', 'train']
	for i,s in enumerate(setName):
		inName = 'pose-files/annotations_master_%s_pascal3d.txt' % s
		oName  = 'pose-files/euler_%s_pascal3d.txt' % s
		inFid  = mpio.GenericWindowReader(inName)
		imDat, lbls = [], []
		N = inFid.num_
		for i in range(inFid.num_):
			im, lb = inFid.read_next()
			imDat.append(im)
			lbls.append(lb)
		inFid.close()
		randSeed = 3 + (2 * i)
		randState = np.random.RandomState(randSeed)
		perm = randState.permutation(N)

		if s == 'train':
			numBad = 2
		else:
			numBad = 0
		oFid = mpio.GenericWindowWriter(oName, N-numBad, 1, 3)
		for p in perm:
			im, lb = imDat[p], lbls[p]
			fName, ch, h, w, x1, y1, x2, y2 = im[0].strip().split()
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			if x2 <= x1 or y2 <= y1:
				print ('Size is weird', x1,x2,y1,y2)
				print ('Skipping', s, im)
				continue
			if x1 <0 or y1<0:
				print ('Too small', x1, y1)
			if x2 > w or y2 > h:
				print ('Too big', x2, w, y2, h)	
			rots = []
			for theta in lb[0]:
				rots.append(su.rot_range(theta)/30.0)
			rots.append(1.0)
			oFid.write(rots, *im)
		oFid.close()


def crop_for_imsize(coords, imSz, padSz=0):
	'''
		get the crop coords so that when the crop is resized to imSz
    then the padding is of size padSz along all borders
	'''
	h, w, x1, y1, x2, y2 = coords
	xSz  = x2 - x1
	ySz  = y2 - y1
	xPad = int(padSz/(float(imSz)/float(xSz)))
	yPad = int(padSz/(float(imSz)/float(ySz)))
	#print (xSz, xPad, ySz, yPad)
	xMn  = max(0, x1 - xPad)
	xMx  = min(w, x2 + xPad)
	yMn  = max(0, y1 - yPad)
	yMx  = min(h, y2 + yPad)
	return xMn, yMn, xMx, yMx, xPad, yPad

def get_filename(fName, isMirror=False):
	prefix = fName.split('.')[0]
	if isMirror:
		return (prefix + '_mirror.jpg')
	else:
		return prefix + '.jpg'
	
def format_raw_label(lb):
	az, el = lb
	az = np.mod(az, 360)
	el = np.mod(el, 360)
	if az > 180:
		az = -(360 - az)
	if el > 180:
		el = -(360 - el)
	az = math.radians(az)
	el = math.radians(el)
	return az, el	

def create_pascal_filestore(imSz=256, padSz=24, debugMode=False):
	dName  = '/data0/pulkitag/data_sets/pascal_3d/imCrop'
	dName = osp.join(dName, 'imSz%d_pad%d_hash') % (imSz, padSz)
	svFile = 'f%s/im%s.jpg' % (imSz, padSz, '%d', '%d')
	srcDir = '/data0/pulkitag/pascal3d/Images' 
	setName = ['train', 'test']
	count, fCount  = 0, 0
	fStore = edict()
	for si, s in enumerate([setName[0]]):
		inName     = 'pose-files/annotations_master_%s_pascal3d.txt' % s
		storeFile  = 'pose-files/pascal3d_dict_%s_imSz%d_pdSz%d.pkl' % (s, imSz, padSz)
		inFid  = mpio.GenericWindowReader(inName)
		imDat, lbls = [], []
		inFid.num_ = inFid.num_ - 1000
		N = inFid.num_
		for i in range(inFid.num_):
			im, lb = inFid.read_next()
			imDat.append(im)
			lbls.append(lb)
		inFid.close()
		randSeed = 7
		randState = np.random.RandomState(randSeed)
		perm = randState.permutation(N)
		if s == 'train':
			numBad = 2
		else:
			numBad = 0
		count = 0
		print (len(perm))
		imList = []
		lbList = []
		for rep, p in enumerate(perm):
			if np.mod(rep,1000)==1:
				print (rep)
			im, lb = imDat[p], lbls[p]
			lb = lb[0]
			fName, ch, h, w, x1, y1, x2, y2 = im[0].strip().split()
			x1, y1, x2, y2, h, w = int(x1), int(y1), int(x2), int(y2), int(h), int(w)
			if x2 <= x1 or y2 <= y1:
				print ('Size is weird', x1,x2,y1,y2)
				print ('Skipping', s, im)
				continue
			if x1 <0 or y1<0:
				print ('Too small', x1, y1)
				continue 
			if x2 > w or y2 > h:
				print ('Too big', x2, w, y2, h)	
				continue
			fPrefix = fName[0:-4]
			svImName = svFile % (fCount, np.mod(count,1000))
			lbFormat = format_raw_label(lb)
			if fPrefix not in fStore.keys():
				fStore[fPrefix] = edict()
				fStore[fPrefix].name   = [svImName]
				fStore[fPrefix].coords = [(x1,y1,x2,y2)]
				fStore[fPrefix].lbs    = [lbFormat]
			else:
				fStore[fPrefix].name.append(svImName)
				fStore[fPrefix].coords.append((x1,y1,x2,y2))
				fStore[fPrefix].lbs.append(lbFormat)
			count += 1
			if np.mod(count,1000) == 0:
				fCount += 1
			#Read and crop the image
			xOg1, yOg1, xOg2, yOg2 = x1, y1, x2, y2
			x1, y1, x2, y2 , xPad, yPad= crop_for_imsize((h, w, x1, y1, x2, y2), imSz, padSz)
			im = scm.imread(osp.join(srcDir, fName))
			if im.ndim == 2:
				im = color.gray2rgb(im)	
			hIm, wIm, chIm = im.shape
			assert hIm==h and wIm==w and chIm==3,(hIm, wIm, chIm, h, w)
			im = cv2.resize(im[y1:y2, x1:x2,:], (imSz, imSz), interpolation=cv2.INTER_LINEAR)
			svImName = osp.join(dName, svImName)
			ou.mkdir(osp.dirname(svImName))
			scm.imsave(svImName, im)
	pickle.dump({'fStore': fStore}, open(storeFile, 'w'))	



def save_my_ass():
	pass	

def create_window_file_v2(imSz=256, padSz=24, debugMode=False):
	dName  = '/data0/pulkitag/data_sets/pascal_3d/imCrop'
	dName = osp.join(dName, 'imSz%d_pad%d_hash/f%s') % (imSz, padSz, '%d')
	svFile = osp.join(dName, 'im%d.jpg')
	srcDir = '/data0/pulkitag/pascal3d/Images' 
	setName = ['train', 'test']
	count, fCount  = 0, 0
	fStore = edict()
	for si, s in enumerate(setName):
		inName = 'pose-files/annotations_master_%s_pascal3d.txt' % s
		oName  = 'pose-files/euler_%s_pascal3d_imSz%d_pdSz%d.txt' % (s, imSz, padSz)
		oFile  = 'pose-files/pascal3d_dict_%s_imSz%d_pdSz%d.pkl' % (s, imSz, padSz)
		inFid  = mpio.GenericWindowReader(inName)
		imDat, lbls = [], []
		N = inFid.num_
		for i in range(inFid.num_):
			im, lb = inFid.read_next()
			imDat.append(im)
			lbls.append(lb)
		inFid.close()
		randSeed = 7
		randState = np.random.RandomState(randSeed)
		perm = randState.permutation(N)
		if s == 'train':
			numBad = 2
		else:
			numBad = 0
		print (len(perm))
		imList = []
		lbList = []
		for rep, p in enumerate(perm):
			if np.mod(rep,1000)==1:
				print (rep, fCount, count)
			im, lb = imDat[p], lbls[p]
			lb = lb[0]
			fName, ch, h, w, x1, y1, x2, y2 = im[0].strip().split()
			x1, y1, x2, y2, h, w = int(x1), int(y1), int(x2), int(y2), int(h), int(w)
			if x2 <= x1 or y2 <= y1:
				print ('Size is weird', x1,x2,y1,y2)
				print ('Skipping', s, im)
				continue
			if x1 <0 or y1<0:
				print ('Too small', x1, y1)
				continue 
			if x2 > w or y2 > h:
				print ('Too big', x2, w, y2, h)	
				continue
			fPrefix = fName[0:-4]
			svImName = svFile % (fCount, np.mod(count,1000))
			if fPrefix not in fStore.keys():
				fStore[fPrefix] = edict()
				fStore[fPrefix].name   = [svImName]
				fStore[fPrefix].coords = [(x1,y1,x2,y2)]
			else:
				fStore[fPrefix].name.append(svImName)
				fStore[fPrefix].coords.append((x1,y1,x2,y2))
			count += 1
			if np.mod(count,1000) == 0:
				fCount += 1
			#Read and crop the image
			xOg1, yOg1, xOg2, yOg2 = x1, y1, x2, y2
			x1, y1, x2, y2 , xPad, yPad= crop_for_imsize((h, w, x1, y1, x2, y2), imSz, padSz)
			im = scm.imread(osp.join(srcDir, fName))
			if im.ndim == 2:
				im = color.gray2rgb(im)	
			hIm, wIm, chIm = im.shape
			assert hIm==h and wIm==w and chIm==3,(hIm, wIm, chIm, h, w)
			im = cv2.resize(im[y1:y2, x1:x2,:], (imSz, imSz), interpolation=cv2.INTER_LINEAR)
			#get filestr
			fStr       = get_filename(svImName)
			fMirrorStr = get_filename(svImName, isMirror=True) 
			svName = osp.join(dName, fStr)
			ou.mkdir(osp.dirname(svName))
			scm.imsave(svName, im)
			#print (svName)
			#pdb.set_trace()
			if debugMode:
				imList.append([fStr, fName, (xOg1, yOg1, xOg2, yOg2), chIm,
            imSz, imSz, 0, 0, imSz, imSz, xPad, yPad])
			else:
				imList.append([fStr, (chIm, imSz, imSz), (0, 0, imSz, imSz)])
			lbList.append(format_raw_label(lb))
			#Mirror the image		
			im = im[:,::-1,:]
			lbMirror = copy.deepcopy(lb)
			#For pascal images, azimuth becomes negative, but elevation doesnot change
			lbMirror[0] = -lb[0]
			svName = osp.join(dName, fMirrorStr)
			scm.imsave(svName, im)
			if debugMode:
				imList.append([fMirrorStr, fName, (xOg1, yOg1, xOg2, yOg2), chIm,
					 imSz, imSz, 0, 0, imSz, imSz, xPad, yPad])
			else:
				imList.append([fMirrorStr, (chIm, imSz, imSz), (0, 0, imSz, imSz)])
			lbList.append(format_raw_label(lbMirror))
		#Write to window file
		N = len(imList)
		perm = randState.permutation(N)
		oFid  = mpio.GenericWindowWriter(oName, N, 1, 2)
		for p in perm:
			oFid.write(lbList[p], imList[p])
		oFid.close()
	if debugMode:
		return imList, lbList


def plot_bbox(bbox, ax=None, drawOpts=None, isShow=True):
	'''
		bbox: x1, y1, x2, y2, conf
					or  x1, y1, x2, y2
	'''
	if isShow:
		plt.ion()
	if ax is None:
		ax = plt.subplot(111)
	if drawOpts is None:
		drawOpts = {'color': 'r', 'linewidth': 3}	
	#Draw the bounding box
	x1, y1, x2, y2, conf = np.floor(bbox)
	ax.plot([x1, x1], [y1, y2], **drawOpts)
	ax.plot([x1, x2], [y2, y2], **drawOpts)
	ax.plot([x2, x2], [y2, y1], **drawOpts)
	ax.plot([x2, x1], [y1, y1], **drawOpts) 
	plt.tight_layout()
	if isShow:	
		plt.draw()
		plt.show()


def vis_im_lb_list(imList, lbList, imSz=256, padSz=24):
	dName  = '/data0/pulkitag/data_sets/pascal_3d/imCrop'
	dName  = osp.join(dName, 'imSz%d_pad%d') % (imSz, padSz)
	srcDir = '/data0/pulkitag/pascal3d/Images' 
	fig    = plt.figure()
	ax     = fig.add_subplot(121)
	ax2    = fig.add_subplot(122)
	plt.ion()
	for iml, lbl in zip(imList, lbList):
		imName, fName, og = iml[0:3]
		x1, y1, x2, y2    = og
		xPad, yPad = iml[-2], iml[-1]
		az, el = format_raw_label(lbl)
		im     = scm.imread(osp.join(dName, imName))
		imFull = scm.imread(osp.join(srcDir, fName)) 
		ax.imshow(im)	
		ax2.imshow(imFull)
		plot_bbox([x1, y1, x2, y2, 0], ax=ax2)	
		plt.suptitle('az: %f, el: %f, xPad: %d, yPad: %d' % (az, el, xPad, yPad))
		plt.draw()
		plt.show()
		ip = raw_input()
		if ip == 'q':
			return	
		plt.cla()

def setup_experiment(isRun=False):
	#Get my best multiloss net
	prms, cPrms = mev2.ptch_pose_euler_smallnet_v5_fc5_exp1_lossl1()
	#lrPrms      = se.get_lr_prms()
	lrPrms       = cPrms.lrPrms
	finePrms    = edict() 
	
	codeDir = '/work4/pulkitag-code/code/projStreetView'	
	finePrms.isSiamese = False
	finePrms['solver'] = lrPrms['solver'] 
	finePrms.paths = edict()
	finePrms.paths.imRootDir  = '/data0/pulkitag/data_sets/pascal_3d-my-copy/PASCAL3D+_release1.1/Images/'
	finePrms.paths.windowFile = edict()
	finePrms.paths.windowFile.train = osp.join(codeDir, 'pose-files/euler_train_pascal3d.txt')
	finePrms.paths.windowFile.test  = osp.join(codeDir, 'pose-files/euler_test_pascal3d.txt')
	#How many layers to finetune
	finePrms.lrAbove = None
	#Jittering
	finePrms.jitter_pct = 0.1
	finePrms.jitter_amt = 0
	#Name of the experiment
	finePrms.expName = 'pascal3d_euler_%s' % lrPrms.expStr 
	exp,modelFile = se.setup_experiment_for_finetune(prms, cPrms, finePrms, 60000) 
	exp.del_layer('ptch_loss')
	exp.del_layer('accuracy')
	exp.del_layer('ptch_fc')
	if isRun:
		exp.make(modelFile=modelFile)
		exp.run()
	
	return exp	
	
