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
import pascal_exp as pep
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

MODULE_PATH = osp.dirname(osp.realpath(__file__))

IM_DATA = []

def get_jitter(coords=None, jitAmt=0, jitPct=0):
	dx, dy = 0, 0
	if jitAmt > 0:
		assert (jitPct == 0)
		rx, ry = np.random.random(), np.random.random()
		dx, dy = rx * jitAmt, ry * jitAmt
		if np.random.random() > 0.5:
			dx = - dx
		if np.random.random() > 0.5:
			dy = -dy
	
	if jitPct > 0:
		h, w = [], []
		for n in range(len(coords)):
			x1, y1, x2, y2 = coords[n]
			h.append(y2 - y1)
			w.append(x2 - x1)
		mnH, mnW = min(h), min(w)
		rx, ry = np.random.random(), np.random.random()
		dx, dy = rx * mnW * jitPct, ry * mnH * jitPct
		if np.random.random() > 0.5:
			dx = - dx
		if np.random.random() > 0.5:
			dy = -dy
	return int(dx), int(dy)	


def image_reader(args):
	imName, imDims, imSz, cropSz, isGray = args
	x1, y1, x2, y2 = imDims
	im = cv2.imread(imName)
	im = cv2.resize(im[y1:y2, x1:x2, :], (imSz, imSz))
	im = im.transpose((2,0,1))
	return im

def image_reader_list(args):
	outList = []
	for ag in args:
		imName, imDims, cropSz, imNum, isGray, isMirror = ag
		x1, y1, x2, y2 = imDims
		im = cv2.imread(imName)
		im = cv2.resize(im[y1:y2, x1:x2, :],
							(cropSz, cropSz))
		if isMirror and np.random.random() >= 0.5:
			im = im[:,::-1,:]
		outList.append((im.transpose((2,0,1)), imNum))
	#glog.info('Processed')
	return outList

def image_reader_scm(args):
	imName, imDims, cropSz, imNum, isGray, isMirror = args
	x1, y1, x2, y2 = imDims
	im = scm.imread(imName)
	im = scm.imresize(im[y1:y2, x1:x2, :],
						(cropSz, cropSz))
	if isMirror and np.random.random() >= 0.5:
		im = im[:,::-1,:]
	im = im[:,:,[2,1,0]].transpose((2,0,1))
	#glog.info('Processed')
	return (im, imNum)


class PascalWindowLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='PythonPascalWindow Layer')
		parser.add_argument('--window_file', default='', type=str)
		parser.add_argument('--im_root_folder', default='', type=str)
		parser.add_argument('--lb_info_file', default='', type=str)
		parser.add_argument('--mean_file', default='None', type=str)
		parser.add_argument('--batch_size', default=128, type=int)
		parser.add_argument('--crop_size', default=192, type=int)
		parser.add_argument('--im_size', default=101, type=int)
		parser.add_argument('--is_gray', dest='is_gray', action='store_true')
		parser.add_argument('--no-is_gray', dest='is_gray', action='store_false')
		parser.add_argument('--resume_iter', default=0, type=int)
		parser.add_argument('--jitter_amt', default=0, type=int)
		parser.add_argument('--ncpu', default=0, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args	

	def __del__(self):
		self.wfid_.close()

	def load_mean(self):
		self.mu_ = None
		if not self.param_.mean_file == 'None':
			#Mean is assumbed to be in BGR format
			self.mu_ = mp.read_mean(self.param_.mean_file)
			self.mu_ = self.mu_.astype(np.float32)
			ch, h, w = self.mu_.shape
			assert (h >= self.param_.crop_size and w >= self.param_.crop_size)
			y1 = int(h/2 - (self.param_.crop_size/2))
			x1 = int(w/2 - (self.param_.crop_size/2))
			y2 = int(y1 + self.param_.crop_size)
			x2 = int(x1 + self.param_.crop_size)
			self.mu_ = self.mu_[:,y1:y2,x1:x2]

	def common_setup(self):
		self.param_ = PascalWindowLayer.parse_args(self.param_str) 
		#Read the window file
		self.wfid_   = mpio.GenericWindowReader(self.param_.window_file)
		self.numIm_  = self.wfid_.numIm_
		self.lblSz_  = self.wfid_.lblSz
		#Check for grayness
		if self.param_.is_gray:
			self.ch_ = 1
		else:
			self.ch_ = 3
		assert self.numIm_ == 1, 'Only 1 image'
		#If needed to resume	
		if self.param_.resume_iter > 0:
			N = self.param_.resume_iter * self.param_.batch_size
			N = np.mod(N, self.wfid_.num_)
			print ('SKIPPING AHEAD BY %d out of %d examples, BECAUSE resume_iter is NOT 0'\
							 % (N, self.wfid_.num_))
			for n in range(N):
				_, _ = self.wfid_.read_next()
		#Function for reading the images	
		if 'cv2' in globals():
			print('OPEN CV FOUND')
			self.readfn_ = image_reader
		else:
			print('OPEN CV NOT FOUND, USING SCM')
			self.readfn_ = image_reader_scm
		#The batch list
		self.argList   = []
		self.labelList = []
		#Load the mean
		self.load_mean()
		#Initialize image data
		self.imData_ = np.zeros((self.param_.batch_size, self.numIm_ * self.ch_,
						self.param_.im_size, self.param_.im_size), np.float32)
		#open the label info file
		lbInfo = pickle.load(open(self.param_.lb_info_file, 'r'))
		self.lbInfo_ = lbInfo['lb']	

	def setup(self, bottom, top):
		pass	
	
	def format_label(self, theta):
		pass
		
	def launch_jobs(self):
		self.argList   = []
		self.labelList = []
		#Form the list of images and labels
		for b in range(self.param_.batch_size):
			if self.wfid_.is_eof():	
				self.wfid_.close()
				self.wfid_   = mpio.GenericWindowReader(self.param_.window_file)
				glog.info('RESTARTING READ WINDOW FILE')
			imNames, lbls = self.wfid_.read_next()
			lb = self.format_label(lbls[0])
			self.labelList.append(lb)
			#Read images
			fName, ch, h, w, x1, y1, x2, y2 = imNames[0].strip().split()
			fName = osp.join(self.param_.im_root_folder, fName)
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			#Computing jittering if required
			#dx, dy = self.get_jitter((x1, y1, x2, y2))
			#Jitter the box
			#x1 = max(0, x1 + dx)
			#y1 = max(0, y1 + dy)
			#x2 = min(w, x2 + dx)
			#y2 = min(h, y2 + dy)
			self.argList.append([fName, (x1,y1,x2,y2), self.param_.im_size, 
         self.param_.crop_size, self.param_.is_gray])
		#Launch the jobs
		if self.param_.ncpu > 0:
			try:
				self.jobs_ = self.pool_.map_async(self.readfn_, argList)
			except KeyboardInterrupt:
				print 'Keyboard Interrupt received - terminating in launch jobs'
				self.pool_.terminate()	

	def get_prefetch_data(self):
		t1 = time.time()
		if self.param_.ncpu > 0:
			try:
				imRes      = self.jobs_.get()
			except:
				print 'Keyboard Interrupt received - terminating'
				self.pool_.terminate()
				raise Exception('Error/Interrupt Encountered')
		else:
			imRes = []
			for b in range(self.param_.batch_size):
				imRes.append(self.readfn_(self.argList[b]))	
		t2= time.time()
		tFetch = t2 - t1
		for b in range(self.param_.batch_size):
			if self.mu_ is not None:	
				self.imData_[b,0:3,:,:] = imRes[b] - self.mu_
			else:
				self.imData_[b,0:3,:,:] = imRes[b]
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


class PascalWindowLayerReg(PascalWindowLayer):
	def setup(self, bottom, top):
		self.common_setup()
		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
								self.param_.im_size, self.param_.im_size)
		#Azimuth cls
		top[1].reshape(self.param_.batch_size, 1, 1, 1)
		#Azimuth bin 1 - reg
		top[2].reshape(self.param_.batch_size, 2, 1, 1)
		#Azimuth bin 2 - reg
		top[3].reshape(self.param_.batch_size, 2, 1, 1)
		#Elevation cls
		top[4].reshape(self.param_.batch_size, 1, 1, 1)
		#Elevation bin 1 - reg
		top[5].reshape(self.param_.batch_size, 2, 1, 1)
		#Elevation bin 2 - reg
		top[6].reshape(self.param_.batch_size, 2, 1, 1)
		#Launch the prefetching	
		self.launch_jobs()
		self.t_ = time.time()	

	def format_theta(self, theta, idx):
		if lbInfo['anglePreProc'] == 'mean_sub':
			mu = lbInfo['mu'][idx]
			sd = None
		theta, flag = pep.format_label(theta, self.lbInfo_, mu=mu, sd=sd)
		if flag == 1:
			return flag, 0., theta
		else:
			return flag, theta, 0.

	def format_label(self, lb):
		lb1 = self.format_theta(lb[0], 0)
		lb2 = self.format_theta(lb[1], 1)	
		return lb1 + lb2
	
	def forward(self):
		t1 = time.time()
		tDiff = t1 - self.t_
		#Load the images
		self.get_prefetch_data()
		top[0].data[...] = self.imData_
		t2 = time.time()
		tFetch = t2-t1
		#Read the labels
		#print self.labelList
		for b in range(self.param_.batch_size):
			lb = self.labelList[b]
			for i in range(6):
				top[i+1].data[b,:,:,:] = 0.
				top[i+1].data[b,0,:,:] = lb[i]
			#Azimuth
			if lb[0] == 0:
				top[2].data[b,1,:,:] = 1.
			else:
				top[3].data[b,1,:,:] = 1.
			#Elevation			
			if lb[3] == 0:
				top[5].data[b,1,:,:] = 1.
			else:
				top[6].data[b,1,:,:] = 1.
	
		self.launch_jobs()
		t2 = time.time()
		#print ('Forward took %fs in PythonWindowDataParallelLayer' % (t2-t1))
		glog.info('Prev: %f, fetch: %f forward: %f' % (tDiff,tFetch, t2-t1))
		self.t_ = time.time()


class PascalWindowLayerCls(PascalWindowLayer):
	def setup(self, bottom, top):
		self.common_setup()
		top[0].reshape(self.param_.batch_size, self.numIm_ * self.ch_,
								self.param_.im_size, self.param_.im_size)
		top[1].reshape(self.param_.batch_size, 2, 1, 1)
		#Launch the prefetching	
		self.launch_jobs()
		self.t_ = time.time()	

	def format_label(self, lb):
		assert self.lbInfo_['anglePreProc'] == 'classify'
		azBin = pep.format_label(lb[0], self.lbInfo_, bins=self.lbInfo_.azBins)
		elBin = pep.format_label(lb[1], self.lbInfo_, bins=self.lbInfo_.elBins)
		return np.array([azBin, elBin]).astype(np.float32)

	def forward(self):
		t1 = time.time()
		tDiff = t1 - self.t_
		#Load the images
		self.get_prefetch_data()
		top[0].data[...] = self.imData_
		t2 = time.time()
		tFetch = t2-t1
		#Read the labels
		#print self.labelList
		for b in range(self.param_.batch_size):
			lb = self.labelList[b]
			top[1].data[b,:,:,:] = lb
		self.launch_jobs()
		t2 = time.time()
		#print ('Forward took %fs in PythonWindowDataParallelLayer' % (t2-t1))
		glog.info('Prev: %f, fetch: %f forward: %f' % (tDiff,tFetch, t2-t1))
		self.t_ = time.time()



def vis_ims():
	protoFile = './python_layers/test/data_layer_pascal_reg.prototxt'
	net = caffe.Net(protoFile, caffe.TEST)
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	plt.ion()
	while True:
		op = net.forward(blobs=['data',
        'az_cls_label', 'az_reg_label_0', 'az_reg_label_1',
        'el_cls_label', 'el_reg_label_0', 'el_reg_label_1'])
		imData = copy.deepcopy(op['data'])
		#lbData = copy.deepcopy(op['label'].squeeze())
		for b in range(imData.shape[0]):
				im = imData[b]
				im = im.transpose((1,2,0))
				im = im[:,:,[2,1,0]].astype(np.uint8)
				ax.imshow(im)	
				#Getting the label
				if op['az_cls_label'][b].squeeze() == 0:
					print (op['az_reg_label_0'][b].squeeze())
					az = -op['az_reg_label_0'][b,0].squeeze()
					assert op['az_reg_label_0'][b,1] == 1.
					assert op['az_reg_label_1'][b,1] == 0.
				else:
					print (op['az_reg_label_1'][b].squeeze())
					az =  op['az_reg_label_1'][b,0].squeeze()
					assert op['az_reg_label_1'][b,1] == 1.
					assert op['az_reg_label_0'][b,1] == 0.
				plt.title('az: %f' % az)
				plt.show()
				plt.draw()
				ip = raw_input()
				if ip == 'q':
					return
				plt.cla()	
