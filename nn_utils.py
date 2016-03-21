import numpy as np
import pickle
import copy

def find_nn(feats1, feats2, numNN=10):
	'''
		for feats1[i] find the NN in feats2
		returns indexes in feats 2
	'''
	idxs = [] 
	for i1 in range(feats1.shape[0]):
		f1   = feats1[i1]
		diff = feats2 - f1
		diff = np.sum(diff * diff,1)
		sortIdx = np.argsort(diff)
		idxs.append(sortIdx[0:numNN])
	return idxs


