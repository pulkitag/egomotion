import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess
import matplotlib.pyplot as plt
import mydisplay as mydisp
import h5py as h5
import pickle
import my_pycaffe_io as mpio
import re
import matplotlib.path as mplPath
import rot_utils as ru
import vis_utils as vu

#Polygon should be of type mplPath	
def show_images(prms, folderId):
	imNames, _ = folderid_to_im_label_files(prms, folderId)	
	plt.ion()
	for imn in imNames:
		im = plt.imread(imn)
		plt.imshow(im)
		inp = raw_input('Press a key to continue')
		if inp=='q':
			return

#Show the groups of images that have the same target point
def show_image_groups(prms, folderId):
	grps = get_target_groups(prms, folderId)
	imNames, lbNames = folderid_to_im_label_files(prms, folderId)
	plt.ion()
	plt.figure()
	for ig, g in enumerate(grps[0:-1]):
		st = g
		en = grps[ig+1]
		print (st,en)
		count = 0
		axl = []
		pltCount = 0
		for i in range(st,en):
			im = plt.imread(imNames[i])
			lb = parse_label_file(lbNames[i])
			if lb.align is not None:
				isAlgn = True
				loc = (lb.align.loc[0], lb.align.loc[1])
				#loc = (lb.align.loc[1], lb.align.loc[0])
			else:
				isAlgn = False
				print ('Align info not found')
				rows, cols, _ = im.shape
				loc = (int(rows/2.0), int(cols/2.0))
			if count < 9:
				ax = plt.subplot(3,3,count+1)
				if isAlgn:
					im = mydisp.box_on_im(im, loc, 27)
				else:
					im = mydisp.box_on_im(im, loc, 27, 'b')
				print im.shape
				ax.imshow(im)
				ax.set_title(('cm: (%.4f, %.4f, %.4f)'
											+ '\n dist: %.4f, head: %.3f, pitch: %.3f, yaw: %3f')\
										% (tuple(lb.pts.camera + [lb.dist] + lb.rots))) 	
				plt.draw()
				axl.append(ax)
				pltCount += 1
			count += 1
		inp = raw_input('Press a key to continue')
		if inp=='q':
			return
		for c in range(pltCount):
			axl[c].cla()

def vis_window_file(setName='test'):
	
