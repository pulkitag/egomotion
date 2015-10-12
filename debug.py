## @package debug
# Debug various aspects of the pipeline
#
import matplotlib as mpl
mpl.use('Agg')
import street_exp as se
import os.path as osp
import street_params as sp
import caffe
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def debug_generic_data():
	#Setup the data proto
	bSz   = 5
	prms  = sp.get_prms_pose(geoFence='dc-v1')	
	nPrms = se.get_nw_prms(imSz=101, netName='smallnet-v2',
								 concatLayer='pool4', maxJitter=0)
	lPrms = se.get_lr_prms(batchsize=bSz)
	cPrms = se.get_caffe_prms(nPrms, lPrms) 
	#Save the data proto
	outFile  = osp.join(prms.paths.baseNetsDr, 'data_debug.prototxt')
	dataDef  = se.make_data_proto(prms, cPrms)
	dataDef.del_layer_property('window_data', 'transform_param', phase='TRAIN')
	dataDef.del_layer_property('window_data', 'transform_param', phase='TEST')
	dataDef.write(outFile)

	print ("Here")
	#Load the data through the layer and save it
	svDr = osp.join(prms.paths.code.dr, 'debug-data')
	plt.ion()
	plt.figure()
	ax1  = plt.subplot(121)
	ax2  = plt.subplot(122)
	print ("Loading Net")
	net = caffe.Net(outFile, caffe.TEST)
	N     = 4
	count = 0
	print ("Processing Data")
	for i in range(N):
		allDat = net.forward(['data', 'data_p'])
		im1Dat = allDat['data']
		im2Dat = allDat['data_p']
		for b in range(bSz):
			print(i,b)
			im1 = im1Dat[b].transpose((1,2,0))
			im2 = im2Dat[b].transpose((1,2,0))
			im1 = im1[:,:,[2,1,0]]
			im2 = im2[:,:,[2,1,0]]
			ax1.imshow(im1.astype(np.uint8))
			ax2.imshow(im2.astype(np.uint8))
			svFile = osp.join(svDr, '%04d.pdf' % count)
			with PdfPages(svFile) as pdf:
				pdf.savefig()
			count += 1
