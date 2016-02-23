## @package street_misc_utils
# Miscellaneos utility functions
#

import numpy as np
from easydict import EasyDict as edict
import os.path as osp
from pycaffe_config import cfg
import os
import pdb
import subprocess
import matplotlib.pyplot as plt
import mydisplay as mydisp
#import h5py as h5
import pickle
import my_pycaffe_io as mpio
import re
import matplotlib.path as mplPath
import rot_utils as ru
from geopy.distance import vincenty as geodist
import copy
import street_params as sp
from multiprocessing import Pool

##
#Convert a pose patch window file into a window file for patch matching only
def convert_pose_ptch_2_ptch(inFile, outFile):
	inFid  = mpio.GenericWindowReader(inFile)
	outFid = mpio.GenericWindowWriter(outFile, inFid.num_, 2, 1)
	while not inFid.is_eof():
		imData, lb = inFid.read_next()
		lbls = [[lb[0][2]]]	
		outFid.write(lbls[0], *imData)
	inFid.close()
	outFid.close()


