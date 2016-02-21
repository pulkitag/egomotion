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

