import caffe
import numpy as np
import argparse, pprint
from multiprocessing import Pool
import scipy.misc as scm
from os import path as osp
import my_pycaffe_io as mpio
import my_pycaffe as mp
from easydict import EasyDict as edict
from transforms3d.transforms3d import euler  as t3eu
import street_label_utils as slu
import other_utils as ou
import pascal_exp as pep
import scipy.io as sio
import time
import glog
import pdb
import pickle
import copy
import matplotlib.pyplot as plt
import math
import pickle
try:
	import cv2
except:
	print('OPEN CV not found, resorting to scipy.misc')
import setup_nyu as snyu

MODULE_PATH = osp.dirname(osp.realpath(__file__))
IM_DATA = []

def get_jitter(xJitAmt, yJitAmt):
	rx, ry = np.random.random(), np.random.random()
	dx, dy = rx * xJitAmt, ry * yJitAmt
	return int(dx), int(dy)	


def transform_image(args):
	im, imDims, imSz = args
	x1, y1, x2, y2 = imDims
	#print (x1, x2, y1, y2)
	im = cv2.resize(im[y1:y2, x1:x2, :], (imSz, imSz))
	im = im.transpose((2,0,1))
	return im

def transform_normals(args):
	nrml, mask, clusters, imDims = args
	x1, y1, x2, y2 = imDims
	nrmls = snyu.normals2cluster(nrml[y1:y2,x1:x2,:], mask[y1:y2,x1:x2], clusters)		
	return nrmls


class NrmlWindowLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='PythonNrmlWindow Layer')
		parser.add_argument('--mean_file', default='None', type=str)
		parser.add_argument('--split', default='None', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--jitter', default=32, type=int)
		parser.add_argument('--crop_scale', default=0.9, type=float)
		parser.add_argument('--im_size', default=101, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

	def __del__(self):
		self.wfid_.close()

	def load_mean(self):
		self.mu_ = None
		if 'binaryproto' in  self.param_.mean_file:
			print ('##### MEAN FILE FOUND ######')
			#Mean is assumbed to be in BGR format
			self.mu_ = mp.read_mean(self.param_.mean_file)
			self.mu_ = self.mu_.astype(np.float32)
			ch, h, w = self.mu_.shape
			assert (h >= self.param_.im_size and w >= self.param_.im_size)
			y1 = int(h/2 - (self.param_.im_size/2))
			x1 = int(w/2 - (self.param_.im_size/2))
			y2 = int(y1 + self.param_.im_size)
			x2 = int(x1 + self.param_.im_size)
			self.mu_ = self.mu_[:,y1:y2,x1:x2]

	def load_nyu_data(self):
		pths        = snyu.get_paths()
		#Load the cluster data
		clusterFile = pths.exp.nrmlClusters
		clData      = pickle.load(open(clusterFile, 'r'))
		self.clusters_ = clData['clusters']
		#Get the splits
		self.idx_ = snyu.get_set_index(self.param_.split)
		print ('NUM Data Points :%d' % len(self.idx_))	
		#Get the mask, images and normals
		self.mask_, self.im_, self.nrml_ = [],[],[] 
		for n in self.idx_:	
			self.mask_.append(snyu.read_mask_from_idx(n))
			self.im_.append(snyu.read_image_from_idx(n))
			self.nrml_.append(snyu.read_normals_from_idx(n))
			 
	def common_setup(self):
		self.param_ = NrmlWindowLayer.parse_args(self.param_str) 
		self.ch_    = 3
		self.numIm_ = 1
		#The batch list
		self.argList   = []
		#Load the mean
		self.load_mean()
		#Load the nyu data
		glog.info('Load NYU Data')
		self.load_nyu_data()
		#Initialize image data
		self.imData_ = np.zeros((self.param_.batch_size, self.numIm_ * self.ch_,
						self.param_.im_size, self.param_.im_size), np.float32)
		#open the label info file
		self.labels_ = np.zeros((self.param_.batch_size, 1,
						20, 20), np.float32)
		#Random state
		self.rand_   = np.random.RandomState(11)
		#Size of the data
		self.imH, self.imW = 426, 560 
		jitH = self.param_.crop_scale * self.imH
		jitW = self.param_.crop_scale * self.imW
		if jitH < self.param_.jitter or jitW < self.param_.jitter:
			glog.info('VERY LARGE JITTER, IMAGES MAY BE RESCALED')

	def setup(self, bottom, top):
		pass	
	
	def format_label(self, theta):
		pass
		
	def launch_jobs(self):
		self.argList   = []
		self.labelList = []
		#Form the list of images and labels
		for b in range(self.param_.batch_size):
			idx        = self.rand_.randint(len(self.idx_))
			xJit, yJit = get_jitter(self.param_.jitter, self.param_.jitter)
			W          = int(self.imW * self.param_.crop_scale)
			H          = int(self.imH * self.param_.crop_scale)
			xCr1, yCr1 = xJit, yJit 
			xCr2  = min(self.imW, xCr1 + W)
			yCr2  = min(self.imH, yCr1 + H)
			self.argList.append([idx, (xCr1,yCr1,xCr2,yCr2), self.param_.im_size])

	def get_prefetch_data(self):
		t1 = time.time()
		for b in range(self.param_.batch_size):
			idx, coords, imSz = self.argList[b]
			im    = transform_image([self.im_[idx], coords, imSz])
			nrmls = transform_normals([self.nrml_[idx], self.mask_[idx],
                   self.clusters_, coords])  
			if self.mu_ is not None:	
				self.imData_[b,0:3,:,:] = im - self.mu_
			else:
				self.imData_[b,0:3,:,:] = im
			self.labels_[b] = nrmls
		t2 = time.time()
		tFetch = t2 - t1
		#print ('%d, Fetching: %f, Copying: %f' % (n, tFetch, time.time()-t2))
		#glog.info('%d, Fetching: %f, Copying: %f' % (n, tFetch, time.time()-t2))
	
	def forward(self, bottom, top):
		pass
		
	def backward(self, top, propagate_down, bottom):
		""" This layer has no backward """
		pass
	
	def reshape(self, bottom, top):
		""" This layer has no reshape """
		pass


class NrmlWindowLayerCls(NrmlWindowLayer):
	def setup(self, bottom, top):
		self.common_setup()
		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
								self.param_.im_size, self.param_.im_size)
		top[1].reshape(self.param_.batch_size, 1, 20, 20)
		glog.info('Setup complete')
		#Launch the prefetching	
		self.launch_jobs()
		self.t_ = time.time()	

	def forward(self, bottom, top):
		t1 = time.time()
		tDiff = t1 - self.t_
		#Load the images
		self.get_prefetch_data()
		top[0].data[...] = self.imData_
		top[1].data[...] = self.labels_
		t2 = time.time()
		tFetch = t2-t1
		self.launch_jobs()
		t2 = time.time()
		glog.info('Prev: %f, fetch: %f forward: %f' % (tDiff,tFetch, t2-t1))
		self.t_ = time.time()


def debug_cls(isPlot=True):
	protoFile = './python_layers/test/data_layer_nrml_cls.prototxt'
	svFile    = './python_layers/test/nrmls/im%d.jpg'
	ou.mkdir(osp.dirname(svFile))
	net   = caffe.Net(protoFile, caffe.TEST)
	count = 0
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	if isPlot:
		plt.ion()
	while True:
		op = net.forward(blobs=['data', 'label'])
		imData = copy.deepcopy(op['data'])
		lbData = copy.deepcopy(op['label'].squeeze())
		for b in range(imData.shape[0]):
				im = imData[b]
				im = im.transpose((1,2,0))
				im = im[:,:,[2,1,0]].astype(np.uint8)
				ax.imshow(im)	
				if isPlot:
					#Getting the label
					#plt.title('az: %f, el: %f' % (lbData[b,0], lbData[b,1]))
					plt.show()
					plt.draw()
				ip = raw_input()
				if ip == 'q':
					return
				if isPlot:
					plt.cla()
				else:
					plt.savefig(svFile % count)
					count += 1	
