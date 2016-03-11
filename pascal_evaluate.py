import scipy.misc as scm
import pascal_exp_run as per
from pascal3d_eval import poseBenchmark as pbench
import cv2

PASCAL_CLS = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
              'chair', 'diningtable', 'motorbike', 'sofa', 'train',
              'tvmomitor']
def get_exp():
	pass

def make_deploy_net(exp, numIter=60000):
	imSz = exp.cPrms_.nwPrms['ipImSz']
	imSz = [[3, imSz, imSz]]
 	exp.make_deploy(dataLayerNames=['window_data'],
      newDataLayerNames=['pair_data'], delAbove='pose_fc',
      imSz=imSz, batchSz=256)
	modelName = exp.get_snapshot_name(numIter=numIter)
	if not osp.exists(modelName):
		print ('ModelFile %s doesnot exist' % modelName)
		return None
	net       = mp.MyNet(exp.files_['netdefDeploy'], modelName)
	#Set preprocessing
	net.set_preprocess(ipName='data', chSwap=None, noTransform=True)
	return net

##
#Get the objec that would be used to generate the benchmark data
def get_bench_obj():
	bench = pbench.PoseBenchmark('car')
	return bench

##
#get images
def get_imdata(imNames, bbox, exp,  padSz=24):
	ims  = []
	imSz = exp.nPrms_.ipImSz 
	for imn, b in zip(imNames, bbox):
		im = cv2.imread(imn)
		x1, y1, x2, y2 = b
		h, w, ch = im.shape
		xMn = max(0, x1 - padSz)
		xMx = min(w, x2 + padSz)
		yMn = max(0, y1 - padSz)
		yMx = max(h, y2 + padSz)
		#Crop
		im = cv2.resize(im[yMn:yMx, xMn:xMx, :], (imSz, imSz))
		#Resize
		#Mean subtaction
		if exp.cPrms_.nwPrms.meanType is None:
			print ('MEAN SUB DONE')
			im  = im.astype(np.float32) - 128.
		else:
			raise Exception('Mean type %s not recognized' % exp.cPrms_.nwPrms.meanType)	
		ims.append(im.reshape((1,) + im.shape))
	ims = np.concatenate(ims)
	return ims


def get_predictions(exp, bench):
	imNames, bbox = bench.giveTestInstances('car')	
	N = len(imNames)
	preds = []
	batchSz = net.get_batchsz()	
	for i in range(0, N, batchSz):
		en  = min(i + batchSz, N)
		ims = get_imdata(imNames[i:en], bbox[i:en])
		if exp.dPrms_.anglePreProc == 'classify'
			pose = net.forward(['azimuth_fc', 'elevation_fc'], **{'data':ims})
			azBin   = copy.deepcopy(pose['azimuth_fc'].squeeze())
			elBin   = copy.deepcopy(pose['elevation_fc'].squeeze())
			for k in range(i, en):
				az    = pep.unformat_label(azBin[k-i], None,
                exp.dPrms_, bins=exp.dPrms_.azBins)
				el    = pep.unformat_label(elBin[k-i], None, 
                exp.dPrms_, bins=exp.dPrms_.elBins)
				preds.append([az, el, 0])
	return preds


def evaluate(exp, bench, preds=None):
	if preds is None:
		preds = get_predictions(exp, bench)
	#Get the ground truth predictions
	gtPose = bench.giveTestPoses('car')
	errs   = bench.evaluatePredictions('car', preds)
	print(np.median(errs))
	return errs    			

