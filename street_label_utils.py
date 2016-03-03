from easydict import EasyDict as edict
import street_config as cfg
import my_exp_config as mec

class LabelPrms(object):
	dbName = cfg.DEF_DB % ('label', 'default') 
	def __init__(self):
		self.lb = edict()
		self.lb['type'] = 'default'

	def get_lbsz(self):
		return 1	

	def get_lbstr(self):
		print (self.dbName)
		return mec.get_sql_id(self.dbName, self.lb)


class PosePrms(LabelPrms):
	dbName = cfg.DEF_DB % ('label', 'pose') 
	def __init__(self, angleType='euler', dof=2):
		LabelPrms.__init__(self)
		self.lb['type']      = 'pose'
		self.lb['angleType'] = angleType
		self.lb['dof']       = dof
		self.lb['maxRot']    = None 

	def get_lbsz(self):
		if self.lb.angleType == 'euler':
			return self.lb.dof
		else:
			raise Exception('Angle Type %s not recognized' % self.lb.angleType)
