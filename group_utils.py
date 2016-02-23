##
#Filter groups by distance
def filter_groups_by_dist(groups, seedGroups, minDist):
	'''
		groups     is a dict
		seedGroups is a dict/list
	'''
	grpKeys = []
	if type(seedGroups) is list:
		itr = enumerate(seedGroups)
	else:
		itr = seedGroups.iteritems()
	for (i,k) in enumerate(groups.keys()):
		#print (i)
		g      = groups[k]
		#Find min distance from all the seed groups
		sgDist = np.inf
		for _,sg in itr:
			#dist = su.get_distance_groups(g, sg)
			dist = su.get_distance_targetpts_groups(g, sg)
			if dist < sgDist:
				sgDist = dist
		if sgDist > minDist:
			grpKeys.append(k)
	#return [sgDist]
	return grpKeys
		
def _filter_groups_by_dist(args):
	return filter_groups_by_dist(*args)

##
#Filter groups by dist parallel
def p_filter_groups_by_dist(prms, grps=None, seedGrps=None):
	numProc = 12
	pool    = Pool(processes=numProc)
	if seedGrps is None:
		seedGrps = su.get_groups(prms, '0052', setName=None)
	if grps is None:
		grps     = su.get_groups(prms, '0048', setName=None)

	if type(grps)==str:
		assert type(seedGrps)==str
		grps     = su.get_groups(prms, grps, setName=None)
		seedGrps = su.get_groups(prms, seedGrps, setName=None)
		
	print (len(seedGrps), len(grps))
	t1 = time.time()
	inArgs = []
	perProc = int(len(grps.keys())/float(numProc))
	count = 0
	grpDict = {}
	L = len(grps.keys())
	for i,gk in enumerate(grps.keys()):
		grpDict[gk] = grps[gk]
		count += 1
		if count == perProc or i == L-1:
			inArgs.append((grpDict, seedGrps, prms.splits.dist))
			count = 0
			grpDict = {}
	try:
		res    = pool.map_async(_filter_groups_by_dist, inArgs)
		pool.close() 
		resKeys = res.get()
	except KeyboardInterrupt:
		pool.terminate()
		raise Exception('Interrupt encountered')

	trKeys = []
	for r in resKeys:
		trKeys = trKeys + r
	t2     = time.time()
	print ("Time: %f" % (t2-t1))
	pool.terminate()
	pool.join()
	return trKeys


##
#Get the groups of images that are taken by pointing at the same
#target location
def get_target_groups(prms, folderId):
	prefixes = get_prefixes(prms, folderId)
	S        = []
	prev     = None
	count    = 0
	for (i,p) in enumerate(prefixes):	
		_,_,_,grp = p.split('_')
		if not(grp == prev):
			S.append(count)
			prev = grp
		count += 1
	return S

##
#Get the groups
def get_groups(prms, folderId, setName='train'):
	'''
		Labels for a particular split
	'''
	grpList   = []
	if prms.geoFence in ['dc-v2', 'cities-v1', 'vegas-v1']:
		keys = get_geo_folderids(prms)
		if folderId not in keys:
			return grpList
		
	if prms.geoFence == 'dc-v1':
		groups = read_geo_groups(prms, folderId)
		gKeys  = groups.keys()
	else:
		#Read labels from the folder
		if prms.isAligned:  
			grpFile = prms.paths.label.grpsAlgn % folderId
		else:
			grpFile = prms.paths.label.grps % folderId
		grpData = pickle.load(open(grpFile,'r'))
		groups  = grpData['groups']
		gKeys   = groups.keys()

	if setName is not None:
		#Find the groups belogning to the split
		splits    = get_train_test_splits(prms, folderId)
		gSplitIds = splits[setName]
		for g in gSplitIds:
			if g in gKeys:
				grpList.append(groups[g])
		return grpList
	else:
		return copy.deepcopy(groups)

##
#Get all the raw labels
def get_groups_all(prms, setName='train'):
	keys = get_folder_keys(prms)
	#keys  = ['0052']
	grps   = []
	for k in keys:
		grps = grps + get_groups(prms, k, setName=setName)
	return grps

##
#Get the overall count of number of groups in the dataset
def get_group_counts(prms):
	dat = pickle.load(open(prms.paths.proc.countFile, 'r'))
	if prms.isAligned:
		keys   = get_folder_keys_aligned(prms)	
	else:
		keys,_ = get_folder_keys_all(prms)	
	count = 0
	for k in keys:
		count += dat['groupCount'][k]
	return count

##
#polygon of type mplPath
def is_geo_coord_inside(polygon,cord):
	return polygon.contains_point(cord)

##
#Find if a group is inside the geofence
def is_group_in_geo(prms, grp):
	isInside = False
	if prms.geoPoly is None:
		return True
	else:
		#Even if a single target point is inside the geo
		#fence count as true
		for geo in prms.geoPoly:
			for i in range(grp.num):
				cc = grp.data[i].pts.target
				isInside = isInside or is_geo_coord_inside(geo, (cc[1], cc[0]))	
	return isInside

##
#Read Geo groups
def read_geo_groups_all(prms):
	geoGrps = edict()
	keys    = get_folder_keys(prms)
	for k in keys:
		geoGrps[k] = read_geo_groups(prms, k)
	return geoGrps

##
#Read geo group from a particular folder
def read_geo_groups(prms, folderId):
	fName      = prms.paths.grp.geoFile % folderId
	data       = pickle.load(open(fName,'r'))
	return data['groups']


##
#Get the distance between groups, 
#Finds the minimum distance between as 
# min(dist_camera_points, dist_target_point)
def get_distance_groups(grp1, grp2):
	minDist = np.inf
	for n1 in range(grp1.num):
		cpt1 = grp1.data[n1].pts.camera[0:2]
		tpt1 = grp1.data[n1].pts.target[0:2]
		for n2 in range(grp2.num):	
			cpt2 = grp2.data[n2].pts.camera[0:2]
			tpt2 = grp2.data[n2].pts.target[0:2]
			cDist = geodist(cpt1, cpt2).meters
			tDist = geodist(tpt1, tpt2).meters
			dist  = min(cDist, tDist)
			if dist < minDist:
				minDist = dist
	return minDist

##
#Seperate points based on the target distance.
def get_distance_targetpts_groups(grp1, grp2):
	tPt1 = grp1.data[0].pts.target
	tPt2 = grp2.data[0].pts.target
	tDist = geodist(tPt1, tPt2).meters
	return tDist

##
#Get the distance between lists of groups
def get_distance_grouplists(grpList1, grpList2):
	minDist = np.inf
	for g1 in grpList1:
		for g2 in grpList2:
			dist = get_distance_groups(g1, g2)
			if dist < minDist:
					minDist = dist
	return minDist	

