import street_exp_v2 as sev2
import street_label_utils as slu
import my_exp_config as mec
import street_config as cfg

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

def simple_euler_dof2_dcv2_smallnetv5(isRun=False,
         gradClip=10, stepsize=20000, base_lr=0.001,
         gamma=0.5, deviceId=0):
	posePrms = slu.PosePrms(maxRot=90, simpleRot=True, dof=2)
	dPrms   =  sev2.get_data_prms(lbPrms=posePrms)
	nwFn    = sev2.process_net_prms
	nwArgs  = {'ncpu': 0, 'baseNetDefProto': 'smallnet-v5_window_siamese_fc5'}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': gradClip,
             'stepsize': stepsize, 'base_lr':base_lr, 'gamma':gamma}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn=sev2.make_net_def, isLog=True)
	if isRun:
		exp.make(deviceId=deviceId)
		exp.run() 
	return exp 	 			


def simple_euler_dof2_dcv2_doublefcv1(isRun=False,
         gradClip=10, stepsize=20000, base_lr=0.001,
         gamma=0.5, deviceId=0):
	posePrms = slu.PosePrms(maxRot=90, simpleRot=True, dof=2)
	dPrms   =  sev2.get_data_prms(lbPrms=posePrms)
	nwFn    = sev2.process_net_prms
	nwArgs  = {'ncpu': 0, 'baseNetDefProto': 'doublefc-v1_window_siamese_fc6'}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': gradClip,
             'stepsize': stepsize, 'base_lr':base_lr, 'gamma':gamma}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn=sev2.make_net_def, isLog=True)
	if isRun:
		exp.make(deviceId=deviceId)
		exp.run() 
	return exp 	 				


